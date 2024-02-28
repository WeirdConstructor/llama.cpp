// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <algorithm>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "common.h"
#include "json.hpp"
#include "llama.h"

#if defined(__unix__) || (defined(__APPLE__) && defined(__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <signal.h>
#include <windows.h>
#endif

#include "grammar-parser.h"

#if defined(_MSC_VER)
#pragma warning(disable : 4244 4267)  // possible loss of data
#endif

using json = nlohmann::json;

static llama_context **g_ctx;
static llama_model **g_model;

static size_t benchmark_start_time;

const char *ws = "\n\r";

// trim from end of string (right)
inline std::string &rtrim_nl(std::string &s, const char *t = ws) {
    s.erase(s.find_last_not_of(t) + 1);
    return s;
}

// trim from beginning of string (left)
inline std::string &ltrim_nl(std::string &s, const char *t = ws) {
    s.erase(0, s.find_first_not_of(t));
    return s;
}

// trim from both ends of string (right then left)
inline std::string &trim_nl(std::string &s, const char *t = ws) {
    return ltrim_nl(rtrim_nl(s, t), t);
}

static std::string tokens_to_output_formatted_string(const llama_context *ctx,
                                                     const llama_token token) {
    std::string out = token == -1 ? "" : llama_token_to_piece(ctx, token);
    // if first bit is 1, meaning it's a partial character
    if (out.size() > 0 && (out[0] & 0x80) == 0x80) {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "\\x" + res;
    }
    return out;
}

static std::string now_timestr() {
    auto t = std::time(nullptr);
    auto tm = *std::localtime(&t);

    std::ostringstream oss;
    oss << std::put_time(&tm, "%d-%m-%Y %H:%M:%S");
    auto str = oss.str();
    return str;
}

struct PromptRunContext {
    int cur_test_nr;
    int total_tests;
    std::string test_id;
    json expected;
    float temp;
    int64_t seed;
    size_t prompt_token_cnt;

    PromptRunContext()
        : cur_test_nr(0),
          total_tests(0),
          temp(1.0),
          seed(0),
          prompt_token_cnt(0) {}
};

json make_token_respose(std::vector<std::string> &responses,
                        PromptRunContext &prc, json tokens);

json make_token_respose(std::vector<std::string> &responses,
                        PromptRunContext &prc, json tokens) {
    if (prc.cur_test_nr <= 0) prc.cur_test_nr = 1;

    int passed_time = time(NULL) - benchmark_start_time;
    float time_per_test = ((float)passed_time) / (float)prc.cur_test_nr;
    float remaining =
        time_per_test * (float)(prc.total_tests - prc.cur_test_nr);
    remaining /= 60.0;

    float passed_time_mins = ((float)passed_time) / 60.0;

    std::string expected_s = "-";
    if (prc.expected.is_string())
        expected_s = prc.expected.template get<std::string>();

    std::string info_str = "[" + std::to_string(prc.cur_test_nr) + "/" +
                           std::to_string(prc.total_tests) +
                           "| id=" + prc.test_id +
                           " #p=" + std::to_string((int)prc.prompt_token_cnt) +
                           " #e=" + expected_s + "]: ";

    for (const auto &tok : tokens) {
        char buf[128];
        float prob = tok[1];
        std::string t = tok[0];
        snprintf(buf, 127, "[%s: %1.4f]", t.c_str(), prob);
        info_str += buf;
    }

    printf("[s/t=%5.2fs, eta=%5.1fm, t=%5.1fm] %s\n", time_per_test, remaining,
           passed_time_mins, info_str.c_str());
    fflush(stdout);

    responses.push_back(prc.test_id + "=" + info_str);

    json single_response;
    single_response["test_id"] = prc.test_id;
    single_response["tokens"] = tokens;
    single_response["expected"] = prc.expected;
    single_response["prompt_token_count"] = (int)prc.prompt_token_cnt;
    single_response["timestamp"] = (int)time(NULL);
    single_response["time"] = now_timestr();

    return single_response;
}

json make_response(std::vector<std::string> &responses, PromptRunContext &prc,
                   const std::string gen, int gen_tok_cnt, json prompt, json sparams_json);

json make_response(std::vector<std::string> &responses, PromptRunContext &prc,
                   const std::string gen, int gen_tok_cnt, json prompt, json sparams_json) {

    std::ostringstream oss;
    oss << std::setprecision(1) << prc.temp;
    std::string temp_str = oss.str();

    if (prc.cur_test_nr <= 0) prc.cur_test_nr = 1;

    int passed_time = time(NULL) - benchmark_start_time;
    float time_per_test = ((float)passed_time) / (float)prc.cur_test_nr;
    float remaining =
        time_per_test * (float)(prc.total_tests - prc.cur_test_nr);
    remaining /= 60.0;

    float passed_time_mins = ((float)passed_time) / 60.0;

    std::string expected_s = "-";
    if (prc.expected.is_string())
        expected_s = prc.expected.template get<std::string>();

    std::string gen_prefix =
        "[" + std::to_string(prc.cur_test_nr) + "/" +
        std::to_string(prc.total_tests) + "| id=" + prc.test_id +
        ", temp=" + temp_str + ", seed=" + std::to_string(prc.seed) +
        ", #p=" + std::to_string((int)prc.prompt_token_cnt) +
        ", #g=" + std::to_string(gen_tok_cnt) + ", #e=" + expected_s + "]:";

    std::string print_gen = gen;
    std::replace(print_gen.begin(), print_gen.end(), '\r', ' ');
    std::replace(print_gen.begin(), print_gen.end(), '\n', '/');
    std::replace(print_gen.begin(), print_gen.end(), '\t', '/');

    printf("[s/t=%5.2fs, eta=%5.1fm, t=%5.1fm] %s %s\n", time_per_test,
           remaining, passed_time_mins, gen_prefix.c_str(), print_gen.c_str());
    fflush(stdout);

    responses.push_back(gen_prefix + gen);

    json single_response;
    single_response["test_id"] = prc.test_id;
    single_response["seed"] = prc.seed;
    single_response["temp"] = temp_str;
    single_response["response"] = gen;
    single_response["expected"] = prc.expected;
    single_response["sampler"] = sparams_json;
    single_response["prompt"] = prompt;
    single_response["prompt_token_count"] = (int)prc.prompt_token_cnt;
    single_response["generated_token_count"] = gen_tok_cnt;
    single_response["timestamp"] = (int)time(NULL);
    single_response["time"] = now_timestr();

    return single_response;
}

struct TextReplacer {
    json prompt_runner_conf;
    json extra_replacements;

    TextReplacer(json prconf) : prompt_runner_conf(prconf) {}

    void add_extra(const std::string &key, const std::string &repl) {
        extra_replacements[key] = repl;
    }

    std::string apply_replacements(json prompt_test, std::string prompt) {
        for (const auto &repl : extra_replacements.items()) {
            std::string search = repl.key();
            std::string replacement = repl.value();
            prompt =
                std::regex_replace(prompt, std::regex(search), replacement);
        }

        if (prompt_runner_conf.find("replacements") !=
            prompt_runner_conf.end()) {
            for (const auto &repl : prompt_runner_conf["replacements"]) {
                std::string search = repl[0];
                std::string replacement = repl[1];
                prompt =
                    std::regex_replace(prompt, std::regex(search), replacement);
            }
        }

        std::string repl_info = "";

        if (prompt_test.find("replacements") != prompt_test.end()) {
            for (const auto &repl : prompt_test["replacements"]) {
                std::string search = repl[0];
                std::string replacement = repl[1];
                if (replacement.size() < 250) {
                    repl_info += search + " := " + replacement + "\n";
                }
                prompt =
                    std::regex_replace(prompt, std::regex(search), replacement);
            }
        }

        if (repl_info.size() > 0) {
            printf("------------------------\n%s", repl_info.c_str());
        }

        return prompt;
    }
};

struct PromptProcessor {
    std::vector<llama_token> embd;
    std::vector<llama_token> embd_gen;
    json stop_sequences;
    int n_past;
    int n_batch;
    struct llama_sampling_context *ctx_sampling;
    int last_response_tok_count;
    std::string last_raw_response;
    bool broken_rep_pen;

    PromptProcessor(int batch, struct llama_sampling_context *ctxs,
                    json prompt_runner_conf)
        : n_past(0),
          n_batch(batch),
          ctx_sampling(ctxs),
          last_response_tok_count(0),
          broken_rep_pen(false) {
        init();
        get_stop_sequences(prompt_runner_conf);
    }

    void set_broken_rep_pen() {
        // due to a bug in the ERP3 benchmark run only the prompt
        // was added to repetition penalty. For backward compat we provide this.
        broken_rep_pen = true;
    }

    void get_stop_sequences(json prompt_runner_conf) {
        if (prompt_runner_conf.find("stop_sequences") !=
            prompt_runner_conf.end()) {
            stop_sequences = prompt_runner_conf["stop_sequences"];
        }
    }

    void init() { llama_kv_cache_clear(*g_ctx); }

    void add_tokens(std::vector<llama_token> tokens) {
        for (auto tok : tokens) {
            embd.push_back(tok);
            llama_sampling_accept(ctx_sampling, *g_ctx, tok, false);
        }
    }

    void add_token(llama_token t) {
        // push the prompt in the sampling context in order
        // to apply repetition penalties later for the
        // prompt, we don't apply grammar rules
        embd.push_back(t);
        if (!broken_rep_pen) {
            llama_sampling_accept(ctx_sampling, *g_ctx, t, false);
        }
    }

    void add_generated(llama_token t) {
        add_token(t);
        embd_gen.push_back(t);
    }

    bool reached_eos() {
        if (!embd.empty() && embd.back() == llama_token_eos(*g_model)) {
            // fprintf(stderr, " [end of text]\n");
            return true;
        }

        return false;
    }

    int get_last_response_token_count() { return last_response_tok_count; }

    std::string get_response(bool without_stop_seq = false,
                             bool trim_trailing = false) {
        last_response_tok_count = 0;
        std::string gen = "";
        for (auto id : embd_gen) {
            last_response_tok_count += 1;
            gen += llama_token_to_piece(*g_ctx, id);
        }
        last_raw_response = gen;

        rtrim_nl(gen);

        if (without_stop_seq) {
            for (const auto &matched : stop_sequences) {
                size_t stop_pos = gen.find(matched);
                if (stop_pos != std::string::npos) {
                    gen = gen.substr(0, stop_pos);
                    break;
                }
            }
        }

        if (trim_trailing) {
            gen = std::regex_replace(
                gen, std::regex("\n\n\n*", std::regex::extended), "\n");
            gen = std::regex_replace(
                gen, std::regex("\\.\\.\\.*", std::regex::extended), "");
            gen = std::regex_replace(
                gen, std::regex("\\*\\*\\**", std::regex::extended), "");
            gen =
                std::regex_replace(gen,
                                   std::regex("(.*[.!?*\")}`$])[^.!?*\")}`$]*",
                                              std::regex::extended),
                                   "$1");
        }

        return gen;
    }

    bool reached_stop_sequence() {
        std::string generated = "";
        for (auto id : embd_gen) {
            generated += llama_token_to_piece(*g_ctx, id);
        }

        for (const auto &matched : stop_sequences) {
            if (generated.find(matched) != std::string::npos) {
                return true;
            }
        }

        return false;
    }

    bool process_tokens() {
        const int n_ctx = llama_n_ctx(*g_ctx);

        if (embd.size() <= 0) {
            return true;
        }

        // Note: n_ctx - 4 here is to match the logic for
        // commandline prompt handling via
        // --prompt or --file which uses the same value.
        auto max_embd_size = n_ctx - 4;
        // Ensure the input doesn't exceed the context size by
        // truncating embd if necessary.
        if ((int)embd.size() > max_embd_size) {
            auto skipped_tokens = embd.size() - max_embd_size;
            printf("<<input too long: skipped %zu token%s>>", skipped_tokens,
                   skipped_tokens != 1 ? "s" : "");
            fflush(stdout);
            embd.resize(max_embd_size);
        }

        // evaluate tokens in batches
        // embd is typically prepared beforehand to fit within a
        // batch, but not always
        for (int i = 0; i < (int)embd.size(); i += n_batch) {
            int n_eval = (int)embd.size() - i;
            if (n_eval > n_batch) {
                n_eval = n_batch;
            }
            if (llama_decode(
                    *g_ctx, llama_batch_get_one(&embd[i], n_eval, n_past, 0))) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return false;
            }
            n_past += n_eval;
        }

        embd.clear();

        return true;
    }
};

void record_token_info(std::vector<std::string> &responses, json &j_tok_resps,
                       PromptRunContext &prc,
                       struct llama_sampling_context *ctx_sampling);

void record_token_info(std::vector<std::string> &responses, json &j_tok_resps,
                       PromptRunContext &prc,
                       struct llama_sampling_context *ctx_sampling) {
    const int n_vocab = llama_n_vocab(*g_model);

    float *logits = llama_get_logits_ith(*g_ctx, 0);

    std::vector<llama_token_data> cur;
    for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
        cur.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
    }

    llama_token_data_array candidates_p = {cur.data(), cur.size(), false};

    llama_sample_grammar(*g_ctx, &candidates_p, ctx_sampling->grammar);

    // Explicitly refuse the " " token.
    for (size_t i = 0; i < candidates_p.size; ++i) {
        const llama_token id = candidates_p.data[i].id;
        const std::string piece = llama_token_to_piece(*g_ctx, id);
        if (piece == " ") {
            candidates_p.data[i].logit = -INFINITY;
        }
    }

    llama_sample_softmax(nullptr, &candidates_p);

    json tokens;

    for (size_t i = 0; i < candidates_p.size; ++i) {
        if (candidates_p.data[i].p > 0.00001) {
            std::string tok = tokens_to_output_formatted_string(
                *g_ctx, candidates_p.data[i].id);
            json j_tok;
            j_tok[0] = tok;
            j_tok[1] = candidates_p.data[i].p;
            tokens.push_back(j_tok);
        }
    }

    if (tokens.size() > 1) {
        j_tok_resps.push_back(make_token_respose(responses, prc, tokens));
    }
}

llama_token generate_sample_seeded(std::vector<std::string> &responses,
                                   json j_resps,
                                   std::vector<int64_t> &sample_seeds,
                                   PromptRunContext &prc,
                                   struct llama_sampling_context *ctx_sampling,
                                   json sparams_json);

llama_token generate_sample_seeded(
    std::vector<std::string> &responses, json j_resps,
    std::vector<int64_t> &sample_seeds, PromptRunContext &prc,
    struct llama_sampling_context *ctx_sampling,
    json sparams_json) {
    llama_token id = 0;
    for (auto seed : sample_seeds) {
        llama_set_rng_seed(*g_ctx, seed);
        prc.seed = seed;

        id = llama_sampling_sample(ctx_sampling, *g_ctx, nullptr);

        if (sample_seeds.size() > 0 && id != llama_token_eos(*g_model)) {
            std::string gen = tokens_to_output_formatted_string(*g_ctx, id);
            j_resps.push_back(make_response(responses, prc, gen, 1, json(), sparams_json));
        }
    }

    return id;
}

bool chatlog_has_repetitions(const std::vector<std::string> &chatlog,
                             std::string &piece, std::string &prevlog);

bool chatlog_has_repetitions(const std::vector<std::string> &chatlog,
                             std::string &piece, std::string &prevlog) {
    if (chatlog.size() < 2) {
        return false;
    }

    std::string last = chatlog[chatlog.size() - 1];
    if (last.size() < 30) {
        return false;
    }

    // if 30 characters appear verbatim in a previous log entry:
    int part_repetition_size = 30;
    if (part_repetition_size > (int)last.size()) {
        part_repetition_size = (int)last.size();
    }
    if (((int)last.size()) > part_repetition_size) {
        // 75% repetition 1:1
        part_repetition_size = ((int)last.size()) * 0.75;
    }

    for (int i = 2; i < (int)chatlog.size(); i++) {
        std::string preventry = chatlog[chatlog.size() - i];
        int diff = ((int)preventry.size()) - (int)last.size();
        if (diff < 0) {
            diff = -1 * diff;
        }
        if (diff > 30) {
            // messages too different in size!
            continue;
        }

        int extra = ((int)last.size()) - part_repetition_size;
        if (extra < 0) {
            extra = 0;
        }
        for (int offs = 0; offs < extra; offs++) {
            std::string subpart = last.substr(offs, part_repetition_size);
            if (preventry.find(subpart) != std::string::npos) {
                piece = subpart;
                prevlog = preventry;
                return true;
            }
        }
    }

    return false;
}

std::string concatl(const std::vector<std::string> &l);

std::string concatl(const std::vector<std::string> &l) {
    std::string logstr;
    for (auto logentry : l) {
        logstr += logentry;
    }
    return logstr;
}

json sparams_to_json(llama_sampling_params &sp);
json sparams_to_json(llama_sampling_params &sp) {
    // clang-format off
    //    int32_t n_prev            = 64;    // number of previous tokens to remember
    //    int32_t n_probs           = 0;     // if greater than 0, output the probabilities of top n_probs tokens.
    //    int32_t top_k             = 40;    // <= 0 to use vocab size
    //    float   top_p             = 0.95f; // 1.0 = disabled
    //    float   tfs_z             = 1.00f; // 1.0 = disabled
    //    float   typical_p         = 1.00f; // 1.0 = disabled
    //    float   temp              = 0.80f; // 1.0 = disabled
    //    int32_t penalty_last_n    = 64;    // last n tokens to penalize (0 = disable penalty, -1 = context size)
    //    float   penalty_repeat    = 1.10f; // 1.0 = disabled
    //    float   penalty_freq      = 0.00f; // 0.0 = disabled
    //    float   penalty_present   = 0.00f; // 0.0 = disabled
    //    int32_t mirostat          = 0;     // 0 = disabled, 1 = mirostat, 2 = mirostat 2.0
    //    float   mirostat_tau      = 5.00f; // target entropy
    //    float   mirostat_eta      = 0.10f; // learning rate
    //    bool    penalize_nl       = true;  // consider newlines as a repeatable token
    // clang-format on

    json j_params;
    j_params["top_k"] = sp.top_k;
    j_params["top_p"] = sp.top_p;
    j_params["min_p"] = sp.min_p;
    j_params["tfs_z"] = sp.tfs_z;
    j_params["typical_p"] = sp.typical_p;
    j_params["temp"] = sp.temp;
    j_params["penalty_present"] = sp.penalty_present;
    j_params["penalty_freq"] = sp.penalty_freq;
    j_params["penalty_last_n"] = sp.penalty_last_n;
    j_params["penalty_repeat"] = sp.penalty_repeat;
    j_params["mirostat"] = sp.mirostat;
    j_params["mirostat_tau"] = sp.mirostat_tau;
    j_params["mirostat_eta"] = sp.mirostat_eta;
    j_params["penalize_nl"] = sp.penalize_nl;
    j_params["order"] = sp.samplers_sequence;
    j_params["order_text"] = llama_sampling_order_print(sp);

    return j_params;
}

int main(int argc, char **argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    llama_sampling_params &sparams = params.sparams;

    if (!sparams.grammar.empty()) {
        grammar_parser::parse_state parsed_grammar;
        parsed_grammar = grammar_parser::parse(sparams.grammar.c_str());

        fprintf(stderr, "%s: grammar:\n", __func__);
        grammar_parser::print_grammar(stderr, parsed_grammar);
        fprintf(stderr, "\n");
        fflush(stderr);
    }

    std::string filename = "prompt_runner_config.json";
    std::ifstream file(filename);
    if (!file) {
        fprintf(stderr,
                "error: failed to open prompt_runner_config config file '%s'\n",
                filename.c_str());
        return 1;
    }

    const json prompt_runner_conf = json::parse(file);

    if (prompt_runner_conf.find("prompt_tests") == prompt_runner_conf.end()) {
        fprintf(stderr, "**********\n");
        fprintf(stderr,
                "ERROR: No prompt_tests in prompt_runner_config.json!\n");
        fprintf(stderr, "**********\n");

        return 1;
    }

    if (params.embedding) {
        printf("\n************\n");
        printf(
            "%s: please use the 'embedding' tool for embedding calculations\n",
            __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.rope_freq_base != 10000.0) {
        fprintf(stderr,
                "%s: warning: changing RoPE frequency base to %g (default "
                "10000.0)\n",
                __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 1.0) {
        fprintf(stderr,
                "%s: warning: scaling RoPE frequency by %g (default 1.0)\n",
                __func__, params.rope_freq_scale);
    }

    LOG_TEE(
        "%s: build = %d (%s)\n", __func__, LLAMA_BUILD_NUMBER, LLAMA_COMMIT);
    LOG_TEE("%s: built with %s for %s\n",
            __func__,
            LLAMA_COMPILER,
            LLAMA_BUILD_TARGET);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    llama_backend_init();

    llama_model *model;
    llama_context *ctx;
    g_model = &model;
    g_ctx = &ctx;

    // load the model and apply lora adapter, if any
    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }
    fprintf(stderr, "loaded model\n");
    fflush(stderr);

    const int n_ctx_train = llama_n_ctx_train(model);
    const int n_ctx = llama_n_ctx(ctx);
    LOG("n_ctx: %d\n", n_ctx);

    if (n_ctx > n_ctx_train) {
        LOG_TEE(
            "%s: warning: model was trained on only %d context tokens (%d "
            "specified)\n",
            __func__, n_ctx_train, n_ctx);
    }

    if (params.n_ctx < 8) {
        fprintf(stderr,
                "%s: warning: minimum context size is 8, using minimum size.\n",
                __func__);
        params.n_ctx = 8;
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(),
                llama_print_system_info());
    }

    // Add BOS if SPM tokenizer
    const bool add_bos = llama_vocab_type(model) == LLAMA_VOCAB_TYPE_SPM;

    // tokenize the prompt
    std::vector<llama_token> embd_inp;
    std::vector<std::string> responses;
    json j_resps;
    json j_tok_resps;

    bool first = true;

    bool record_next_token_info =
        prompt_runner_conf.value("record_next_token_info", false);

    std::vector<int64_t> sample_seeds;
    if (prompt_runner_conf.find("sample_seeds") != prompt_runner_conf.end()) {
        fprintf(stderr, "Reading sample_seeds\n");
        for (const auto &seed_value : prompt_runner_conf["sample_seeds"]) {
            int64_t seed = seed_value;
            sample_seeds.push_back(seed);
        }
    }

    std::vector<float> temps;

    if (prompt_runner_conf.find("temps") != prompt_runner_conf.end()) {
        fprintf(stderr, "Reading temps\n");
        for (const auto &temp : prompt_runner_conf["temps"]) {
            temps.push_back(temp);
        }
    } else {
        temps.push_back(sparams.temp);
    }

    std::vector<int64_t> seeds;
    if (prompt_runner_conf.find("seeds") != prompt_runner_conf.end()) {
        fprintf(stderr, "Reading seeds\n");
        for (const auto &seed_value : prompt_runner_conf["seeds"]) {
            int64_t seed = seed_value;
            printf("seed value: %ld\n", seed);
            seeds.push_back(seed);
        }
    } else {
        seeds.push_back(params.seed);
    }

    int test_count = 0;
    if (prompt_runner_conf.find("prompt_tests") != prompt_runner_conf.end()) {
        test_count = prompt_runner_conf["prompt_tests"].size();
    } else {
        fprintf(
            stderr,
            "No \"prompt_tests\" defined in the prompt_runner_config.json!\n");
        return 1;
    }

    benchmark_start_time = time(NULL);

    fprintf(stderr, "PROMPT-RUNNER-START\n");
    fflush(stderr);

    TextReplacer replacer(prompt_runner_conf);

    PromptRunContext prun_ctx;
    prun_ctx.total_tests = seeds.size() * temps.size() * test_count;

    for (const auto &prompt_test : prompt_runner_conf["prompt_tests"]) {
        json expected;
        if (prompt_test.find("expected") != prompt_test.end()) {
            expected = prompt_test["expected"];
        }
        prun_ctx.expected = expected;

        prun_ctx.test_id = prompt_test.value("id", "unknown_test_id");

        std::string prompt =
            replacer.apply_replacements(prompt_test, params.prompt);
        rtrim_nl(prompt);

        if (params.verbose_prompt) {
            printf(
                "PROMPT------------------------\n%s\n--------------------------"
                "---\n",
                prompt.c_str());
        }

        fflush(stdout);

        for (auto &temp : temps) {
            sparams.temp = temp;
            prun_ctx.temp = temp;

            json sparams_json = sparams_to_json(sparams);

            for (const auto &seed_value : seeds) {
                prun_ctx.seed = seed_value;
                llama_set_rng_seed(ctx, prun_ctx.seed);

                prun_ctx.cur_test_nr += 1;

                if (params.prompt.empty()) {
                    fprintf(stderr, "No prompt given!");
                    return 1;
                }

                embd_inp = ::llama_tokenize(ctx, prompt.c_str(), add_bos, true);

                const int n_ctx = llama_n_ctx(ctx);
                prun_ctx.prompt_token_cnt = embd_inp.size();

                if (((int)embd_inp.size() + (int)params.n_predict) >
                    n_ctx - 4) {
                    fprintf(stderr,
                            "%s: error: prompt is too long (%d tokens, %d "
                            "predict, max %d)\n",
                            __func__, (int)embd_inp.size(),
                            (int)params.n_predict, n_ctx - 4);
                    return 1;
                }

                //                if (params.verbose_prompt) {
                //                    fprintf(stderr, "\n");
                //                    fprintf(stderr, "%s: prompt: '%s'\n",
                //                    __func__,
                //                            params.prompt.c_str());
                //                    fprintf(stderr, "%s: number of tokens in
                //                    prompt = %zu\n",
                //                            __func__, embd_inp.size());
                //                    for (int i = 0; i < (int)embd_inp.size();
                //                    i++) {
                //                        fprintf(stderr, "%6d -> '%s'\n",
                //                        embd_inp[i],
                //                                llama_token_to_piece(ctx,
                //                                embd_inp[i]).c_str());
                //                    }
                //
                //                    fprintf(stderr, "\n");
                //                }

                if (first) {
                    fprintf(stderr, "sampling: \n%s\n",
                            llama_sampling_print(sparams).c_str());
                }

                if (prompt_test.find("chat") != prompt_test.end()) {
                    json chat = prompt_test["chat"];

                    json user_prompt;
                    if (chat.contains("user")) {
                        user_prompt = chat["user"];
                    } else {
                        fprintf(stderr, "BAD CHAT, NO \"user\" KEY!\n");
                        return 1;
                    }

                    json char_prompt;
                    if (chat.contains("char")) {
                        char_prompt = chat["char"];
                    } else {
                        fprintf(stderr, "BAD CHAT, NO \"user\" KEY!\n");
                        return 1;
                    }

                    printf("FOOO\n");
                    int chat_turns = chat.value("turns", 3);
                    printf("FOOO2\n");

                    std::vector<std::string> chatlog;

                    std::string char_log_init = chat.value("char_log_init", "");
                    if (char_log_init.size() > 0) {
                        replacer.add_extra("<RESPONSE>", char_log_init);
                        std::string log_fmt =
                            char_prompt.value("log_fmt", "<RESPONSE>");
                        char_log_init =
                            replacer.apply_replacements(prompt_test, log_fmt);
                        chatlog.push_back(char_log_init);
                    }

                    bool is_char_turn = false;
                    int gen_token_count_sum = 0;
                    json raw_chatlog;
                    bool ended_with_empty = false;
                    bool repeated_itself = false;
                    std::string repeated_part;
                    std::string repeated_log_entry;
                    std::string repeated_current_entry;

                    json last_char_prompt;
                    json last_user_prompt;

                    for (int turn_i = 0; turn_i < chat_turns; turn_i++) {
                        std::string whose_response = "user";
                        json chat_prompt = user_prompt;
                        if (is_char_turn) {
                            whose_response = "char";
                            chat_prompt = char_prompt;
                        }

                        int n_remain = chat_prompt.value("n_gen", 100);
                        std::string prompt =
                            chat_prompt.value("prompt", "NO CHAT PROMPT");
                        std::string log_fmt =
                            chat_prompt.value("log_fmt", "<RESPONSE>");

                        printf("##################################\n");
                        std::string logstr = concatl(chatlog);
                        printf("LOG:\n%s", logstr.c_str());
                        fflush(stdout);

                        replacer.add_extra("<CHATLOG>", logstr);
                        prompt =
                            replacer.apply_replacements(prompt_test, prompt);
                        rtrim_nl(prompt);

                        // d// printf("FOOO5[%s]\n", prompt.c_str());
                        std::vector<llama_token> chat_embd_inp;
                        chat_embd_inp = ::llama_tokenize(ctx, prompt.c_str(),
                                                         add_bos, true);
                        printf("########( PLEN: %ld )##############\n",
                               chat_embd_inp.size());

                        json prompt_info;
                        prompt_info["tokens"] = (int)chat_embd_inp.size();
                        prompt_info["prompt"] = prompt;

                        if (is_char_turn) {
                            last_char_prompt = prompt_info;
                        } else {
                            last_user_prompt = prompt_info;
                        }
                        is_char_turn = !is_char_turn;

                        prun_ctx.prompt_token_cnt = chat_embd_inp.size();

                        struct llama_sampling_context *ctx_sampling =
                            llama_sampling_init(sparams);

                        PromptProcessor proc(params.n_batch, ctx_sampling,
                                             prompt_runner_conf);

                        if (prompt_test.value("broken_rep_pen", false)) {
                            proc.set_broken_rep_pen();
                        }

                        proc.add_tokens(chat_embd_inp);

                        while (n_remain > 0) {
                            if (!proc.process_tokens()) {
                                return 1;
                            }

                            llama_token id = llama_sampling_sample(
                                ctx_sampling, ctx, nullptr);

                            proc.add_generated(id);
                            --n_remain;

                            if (proc.reached_eos()) {
                                break;
                            }

                            if (proc.reached_stop_sequence()) {
                                n_remain = 0;
                                break;
                            }
                        }

                        std::string gen = proc.get_response(true, true);
                        int gen_tok_cnt = proc.get_last_response_token_count();

                        json logentry;
                        logentry.push_back(whose_response);
                        logentry.push_back(gen_tok_cnt);
                        logentry.push_back(proc.last_raw_response);
                        raw_chatlog.push_back(logentry);

                        gen_token_count_sum += gen_tok_cnt;
                        trim_nl(gen, " \r\n");
                        if (gen.size() == 0) {
                            ended_with_empty = true;
                            printf(
                                "EMPTY "
                                "RESPONSE!\n################################\n%"
                                "s\n*******************************************"
                                "************\n",
                                prompt.c_str());
                        }
                        replacer.add_extra("<RESPONSE>", gen);
                        std::string new_log =
                            replacer.apply_replacements(prompt_test, log_fmt);
                        fflush(stdout);
                        chatlog.push_back(new_log);
                        if (chatlog_has_repetitions(chatlog, repeated_part,
                                                    repeated_log_entry)) {
                            repeated_current_entry = new_log;
                            repeated_itself = true;
                            break;
                        }

                        llama_sampling_free(ctx_sampling);

                        if (gen.size() == 0) {
                            break;
                        }
                    }

                    json prompt_collection;
                    prompt_collection["char"] = last_char_prompt;
                    prompt_collection["user"] = last_user_prompt;
                    prompt_collection["raw_chatlog"] = raw_chatlog;
                    json end_reason;
                    end_reason["empty_response"] = ended_with_empty;
                    end_reason["repeated_response"] = repeated_itself;
                    if (repeated_itself) {
                        end_reason["repeated_part"] = repeated_part;
                        end_reason["repeated_log_entry"] = repeated_log_entry;
                        end_reason["repeated_current_entry"] =
                            repeated_current_entry;
                    }
                    prompt_collection["end_reason"] = end_reason;

                    j_resps.push_back(
                        make_response(responses, prun_ctx, concatl(chatlog),
                                      gen_token_count_sum, prompt_collection, sparams_json));

                } else {
                    struct llama_sampling_context *ctx_sampling =
                        llama_sampling_init(sparams);

                    PromptProcessor proc(params.n_batch, ctx_sampling,
                                         prompt_runner_conf);
                    if (prompt_test.value("broken_rep_pen", false)) {
                        proc.set_broken_rep_pen();
                    }
                    proc.add_tokens(embd_inp);

                    int n_remain = params.n_predict;
                    while (n_remain > 0) {
                        if (!proc.process_tokens()) {
                            return 1;
                        }

                        if (record_next_token_info) {
                            record_token_info(responses, j_tok_resps, prun_ctx,
                                              ctx_sampling);
                        }

                        llama_token id = 0;
                        if (sample_seeds.size() > 0) {
                            id = generate_sample_seeded(responses, j_resps,
                                                        sample_seeds, prun_ctx,
                                                        ctx_sampling, sparams_json);
                        } else {
                            id = llama_sampling_sample(ctx_sampling, ctx,
                                                       nullptr);
                        }

                        proc.add_generated(id);
                        --n_remain;

                        if (proc.reached_eos()) {
                            break;
                        }

                        if (proc.reached_stop_sequence()) {
                            n_remain = 0;
                            break;
                        }
                    }

                    std::string gen = proc.get_response();
                    int gen_tok_cnt = proc.get_last_response_token_count();

                    j_resps.push_back(make_response(responses, prun_ctx, gen,
                                                    gen_tok_cnt, json(), sparams_json));

                    llama_sampling_free(ctx_sampling);
                }

                first = false;
            }
        }
    }

    llama_print_timings(ctx);

    std::string model_file = params.model.c_str();
    model_file = model_file.substr(model_file.find_last_of("/\\") + 1);
    printf("model: %s\n", model_file.c_str());

    for (auto resp : responses) {
        printf("%s\n", resp.c_str());
    }

    std::string responses_json_dump =
        j_resps.dump(2, ' ', false, json::error_handler_t::replace);
    printf("%s\n", responses_json_dump.c_str());
    fflush(stdout);

    json j_params;
    j_params["rope_freq_base"] = params.rope_freq_base;
    j_params["rope_freq_scale"] = params.rope_freq_scale;
    j_params["temp"] = sparams.temp;
    j_params["top_k"] = sparams.top_k;
    j_params["top_p"] = sparams.top_p;
    j_params["tfs_z"] = sparams.tfs_z;
    j_params["typical_p"] = sparams.typical_p;
    j_params["repeat_last_n"] = sparams.penalty_last_n;
    j_params["repeat_penality"] = sparams.penalty_repeat;

    json build_info;
    build_info["build"] = LLAMA_BUILD_NUMBER;
    build_info["commit"] = LLAMA_COMMIT;
    build_info["compiler"] = LLAMA_COMPILER;
    build_info["build_target"] = LLAMA_BUILD_TARGET;

    json results;
    results["llama_cpp_build_info"] = build_info;
    results["params"] = j_params;
    results["model_file"] = model_file;
    results["prompt"] = std::string(params.prompt);
    results["config"] = prompt_runner_conf;
    results["results"] = j_resps;
    if (record_next_token_info) {
        results["tokens"] = j_tok_resps;
    }

    std::string out_file_name =
        "result_" + std::to_string(time(NULL)) + "_" + model_file + ".json";
    std::ofstream outf(out_file_name);
    outf << results.dump(2, ' ', false, json::error_handler_t::replace);
    outf.close();

    printf("[PROMPT_RUNNER_OUTPUT_FILE: %s]\n", out_file_name.c_str());
    fflush(stdout);

    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
