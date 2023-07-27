// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common.h"
#include "llama.h"
#include "build-info.h"
#include "grammar-parser.h"

#include "json.hpp"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <regex>
#include <filesystem>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

#if defined(_MSC_VER)
#pragma warning(disable: 4244 4267) // possible loss of data
#endif

using json = nlohmann::json;

static llama_context ** g_ctx;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    if (signo == SIGINT) {
        printf("\n");
        llama_print_timings(*g_ctx);
        _exit(130);
    }
}
#endif

const char* ws = "\n\r";

// trim from end of string (right)
inline std::string& rtrim_nl(std::string& s, const char* t = ws)
{
    s.erase(s.find_last_not_of(t) + 1);
    return s;
}

// trim from beginning of string (left)
inline std::string& ltrim_nl(std::string& s, const char* t = ws)
{
    s.erase(0, s.find_first_not_of(t));
    return s;
}

// trim from both ends of string (right then left)
inline std::string& trim_nl(std::string& s, const char* t = ws)
{
    return ltrim_nl(rtrim_nl(s, t), t);
}

static std::string tokens_to_output_formatted_string(const llama_context *ctx, const llama_token token)
{
    std::string out = token == -1 ? "" : llama_token_to_str(ctx, token);
    // if first bit is 1, meaning it's a partial character
    if (out.size() > 0 && (out[0] & 0x80) == 0x80)
    {
        std::stringstream ss;
        ss << std::hex << (out[0] & 0xff);
        std::string res(ss.str());
        out = "\\x" + res;
    }
    return out;
}

json make_response(std::vector<std::string> &responses, int cur_test_nr,
    int total_tests, const std::string &test_id, float temp,
    int64_t seed, const std::vector<llama_token> &embd_gen)
{
    llama_context *ctx = *g_ctx;

    std::ostringstream oss;
    oss << std::setprecision(1) << temp;
    std::string temp_str = oss.str();

    std::string gen_prefix =
        "[" + std::to_string(cur_test_nr) + "/" + std::to_string(total_tests)
        + "| id=" + test_id
        + ", temp=" + temp_str
        + ", seed=" + std::to_string(seed)
        + "]:";

    std::string gen = "";
    for (auto id : embd_gen) {
        gen += tokens_to_output_formatted_string(ctx, id);
    }

    printf("%s %s\n", gen_prefix.c_str(), gen.c_str());
    fflush(stdout);

    responses.push_back(gen_prefix + gen);

    json single_response;
    single_response["test_id"] = test_id;
    single_response["seed"] = seed;
    single_response["temp"] = temp_str;
    single_response["response"] = gen;

    return single_response;
}

int process_prompt(const std::string &prompt, const gpt_params &params, std::vector<llama_token> &last_n_tokens) {
    llama_context *ctx = *g_ctx;

    const int n_ctx = llama_n_ctx(ctx);

    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

    std::vector<llama_token> embd_inp =
        ::llama_tokenize(ctx, prompt.c_str(), true);

    // Note: n_ctx - 4 here is to match the logic for commandline prompt handling via
    auto max_embd_size = n_ctx - 4;

    // Ensure the input doesn't exceed the context size by truncating embd if necessary.
    if ((int)embd_inp.size() > max_embd_size) {
        auto skipped_tokens = embd_inp.size() - max_embd_size;
        printf("<<input too long: skipped %zu token%s>>",
            skipped_tokens, skipped_tokens != 1 ? "s" : "");
        fflush(stdout);
        embd_inp.resize(max_embd_size);
    }

    if (params.verbose_prompt) {
        fprintf(stderr, "\n");
        fprintf(stderr, "%s: prompt: '%s'\n", __func__, params.prompt.c_str());
        fprintf(stderr, "%s: number of tokens in prompt = %zu\n",
            __func__, embd_inp.size());
        for (int i = 0; i < (int) embd_inp.size(); i++) {
            fprintf(stderr, "%6d -> '%s'\n",
                embd_inp[i], llama_token_to_str(ctx, embd_inp[i]));
        }

        fprintf(stderr, "\n");
    }

    int n_past = 0;

    for (int i = 0; i < (int) embd_inp.size(); i += params.n_batch) {
        int n_eval = (int) embd_inp.size() - i;
        if (n_eval > params.n_batch) {
            n_eval = params.n_batch;
        }

        printf("### PROC: i=%d, n_eval=%d, n_past=%d\n", i, n_eval, n_past);

        if (llama_eval(ctx, &embd_inp[i], n_eval, n_past, params.n_threads)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return -1;
        }

        for (int j = i; j < (i + n_eval); j++) {
            last_n_tokens.erase(last_n_tokens.begin());
            last_n_tokens.push_back(embd_inp[j]);
        }

        n_past += n_eval;
    }

    return n_past;
}

int main(int argc, char ** argv) {
    gpt_params params;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    std::string filename = "prompt_runner_config.json";
    std::ifstream file(filename);
    if (!file) {
        fprintf(stderr, "error: failed to open prompt_runner_config config file '%s'\n", filename.c_str());
        return 1;
    }

    const json prompt_runner_conf = json::parse(file);

    if (params.perplexity) {
        printf("\n************\n");
        printf("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.embedding) {
        printf("\n************\n");
        printf("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        printf("************\n\n");

        return 0;
    }

    if (params.rope_freq_base != 10000.0) {
        fprintf(stderr, "%s: warning: changing RoPE frequency base to %g (default 10000.0)\n", __func__, params.rope_freq_base);
    }

    if (params.rope_freq_scale != 1.0) {
        fprintf(stderr, "%s: warning: scaling RoPE frequency by %g (default 1.0)\n", __func__, params.rope_freq_scale);
    }

    if (params.n_ctx > 2048) {
        // TODO: determine the actual max context of the model (e.g. 4096 for LLaMA v2) and use that instead of 2048
        fprintf(stderr, "%s: warning: base model only supports context sizes no greater than 2048 tokens (%d specified)\n", __func__, params.n_ctx);
    } else if (params.n_ctx < 8) {
        fprintf(stderr, "%s: warning: minimum context size is 8, using minimum size.\n", __func__);
        params.n_ctx = 8;
    }

    fprintf(stderr, "%s: build = %d (%s)\n", __func__, BUILD_NUMBER, BUILD_COMMIT);

    if (params.seed == LLAMA_DEFAULT_SEED) {
        params.seed = time(NULL);
    }

    fprintf(stderr, "%s: seed  = %u\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }

    llama_backend_init(params.numa);

    llama_model *model = nullptr;
    llama_context *ctx = nullptr;
    g_ctx = &ctx;

    // load the model and apply lora adapter, if any
    auto lparams = llama_context_params_from_gpt_params(params);
    std::tie(model, ctx) = llama_init_from_gpt_params(params);

    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
    }

    grammar_parser::parse_state parsed_grammar;
    if (!params.grammar.empty()) {
        parsed_grammar = grammar_parser::parse(params.grammar.c_str());
        // will be empty (default) if there are parse errors
        if (parsed_grammar.rules.empty()) {
            return 1;
        }

        fprintf(stderr, "%s: grammar:\n", __func__);
        grammar_parser::print_grammar(stderr, parsed_grammar);
        fprintf(stderr, "\n");
    }

    // Add a space in front of the first character to match OG llama tokenizer behavior
    params.prompt.insert(0, 1, ' ');

    // do one empty run to warm up the model
    {
        const std::vector<llama_token> tmp = { llama_token_bos(), };
        llama_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);
        llama_reset_timings(ctx);
    }

    std::vector<std::string> responses;
    json j_resps;

    bool first = true;

    std::vector<int64_t> sample_seeds;
    if (prompt_runner_conf.find("sample_seeds") != prompt_runner_conf.end()) {
        for (const auto &seed_value : prompt_runner_conf["sample_seeds"]) {
            int64_t seed = seed_value;
            sample_seeds.push_back(seed);
        }
    }

    std::vector<float> temps;

    if (prompt_runner_conf.find("temps") != prompt_runner_conf.end()) {
        for (const auto &temp : prompt_runner_conf["temps"]) {
            temps.push_back(temp);
        }
    } else {
        temps.push_back(params.temp);
    }

    std::vector<int64_t> seeds;
    if (prompt_runner_conf.find("seeds") != prompt_runner_conf.end()) {
        for (const auto &seed_value : prompt_runner_conf["seeds"]) {
            int64_t seed = seed_value;
            printf("seed value: %ld\n", seed);
            seeds.push_back(seed);
        }
    } else {
        seeds.push_back(params.seed);
    }

    int total_tests =
        seeds.size() * temps.size() * prompt_runner_conf["prompt_tests"].size();
    int current_test_nr = 0;


    for (const auto &prompt_test : prompt_runner_conf["prompt_tests"]) {
        std::string prompt = params.prompt;

        std::string test_id = prompt_test["id"];

        for (const auto &repl : prompt_runner_conf["replacements"]) {
            std::string search = repl[0];
            std::string replacement = repl[1];
            prompt = std::regex_replace(prompt, std::regex(repl[0]), replacement);
        }

        for (const auto &repl : prompt_test["replacements"]) {
            std::string search = repl[0];
            std::string replacement = repl[1];
            prompt = std::regex_replace(prompt, std::regex(repl[0]), replacement);
        }

        rtrim_nl(prompt);

        // TODO: replace with ring-buffer
        const int n_ctx = llama_n_ctx(ctx);
        std::vector<llama_token> last_n_tokens_cached(n_ctx);
        std::fill(last_n_tokens_cached.begin(), last_n_tokens_cached.end(), 0);

        int n_past_cached = process_prompt(prompt, params, last_n_tokens_cached);

        // TODO WEICON:
        // - save state to memory here, destroy context

        const size_t n_state_size_max = llama_get_state_size(ctx);
        uint8_t *state_mem = new uint8_t[n_state_size_max];
        /* const size_t n_state_size_cur = */ llama_copy_state_data(ctx, state_mem);

        llama_free(ctx);
        ctx = nullptr;

        printf("PROMPT------------------------\n%s\n-----------------------------\n", prompt.c_str());
        for (auto &temp : temps) {
            params.temp = temp;

            for (const auto &seed_value : seeds) {
                int64_t seed = seed_value;
                // TODO WEICON:
                // - make new context with old lparams (save them somehow?!)
                // - restore context from memory, that was saved above.
                // - 
                ctx = llama_new_context_with_model(model, lparams);
                llama_set_state_data(ctx, state_mem);
                llama_set_rng_seed(ctx, seed);

                std::vector<llama_token> last_n_tokens = last_n_tokens_cached;
                printf("LAST N TOKEN: %d\n", (int) last_n_tokens_cached[last_n_tokens_cached.size() - 1]);
                int n_past = n_past_cached;
                int n_remain = params.n_predict;

                current_test_nr += 1;

                if (first) {
                    fprintf(stderr, "sampling: repeat_last_n = %d, repeat_penalty = %f, presence_penalty = %f, frequency_penalty = %f, top_k = %d, tfs_z = %f, top_p = %f, typical_p = %f, temp = %f, mirostat = %d, mirostat_lr = %f, mirostat_ent = %f\n",
                            params.repeat_last_n, params.repeat_penalty, params.presence_penalty, params.frequency_penalty, params.top_k, params.tfs_z, params.top_p, params.typical_p, params.temp, params.mirostat, params.mirostat_eta, params.mirostat_tau);
                    fprintf(stderr, "generate: n_ctx = %d, n_batch = %d, n_predict = %d\n", n_ctx, params.n_batch, params.n_predict);
                    fprintf(stderr, "\n\n");
                }

                llama_grammar *grammar = NULL;
                if (!params.grammar.empty()) {
                    {
                        auto it = params.logit_bias.find(llama_token_eos());
                        if (it != params.logit_bias.end() && it->second == -INFINITY) {
                            fprintf(stderr,
                                "%s: warning: EOS token is disabled, which will cause most grammars to fail\n", __func__);
                        }
                    }

                    std::vector<const llama_grammar_element *> grammar_rules(parsed_grammar.c_rules());
                    grammar = llama_grammar_init(
                        grammar_rules.data(), grammar_rules.size(), parsed_grammar.symbol_ids.at("root"));
                }

                std::vector<llama_token> embd;
                std::vector<llama_token> embd_gen;

                while (n_remain != 0) {
                    printf("n_remain=%d embd.size=%d\n", n_remain, (int) embd.size());
                    // predict
                    if (embd.size() > 0) {
                        // evaluate tokens in batches
                        for (int i = 0; i < (int) embd.size(); i += params.n_batch) {
                            int n_eval = (int) embd.size() - i;
                            if (n_eval > params.n_batch) {
                                n_eval = params.n_batch;
                            }

                            printf("### PROC[%d]: i=%d, n_eval=%d, n_past=%d\n", embd[i], i, n_eval, n_past);
                            if (llama_eval(ctx, &embd[i], n_eval, n_past, params.n_threads)) {
                                fprintf(stderr, "%s : failed to eval\n", __func__);
                                return 1;
                            }

                            n_past += n_eval;
                        }
                    }

//                    std::string gen = "";
//                    for (auto id : embd) {
//                        gen += llama_token_to_str(ctx, id);
//                    }
//                    printf("[[%s]]\n", gen.c_str());
//                    fflush(stdout);

                    embd.clear();

                    {
                        // out of user input, sample next token
                        const float   temp            = params.temp;
                        const int32_t top_k           = params.top_k <= 0 ? llama_n_vocab(ctx) : params.top_k;
                        const float   top_p           = params.top_p;
                        const float   tfs_z           = params.tfs_z;
                        const float   typical_p       = params.typical_p;
                        const int32_t repeat_last_n   = params.repeat_last_n < 0 ? n_ctx : params.repeat_last_n;
                        const float   repeat_penalty  = params.repeat_penalty;
                        const float   alpha_presence  = params.presence_penalty;
                        const float   alpha_frequency = params.frequency_penalty;
                        const int     mirostat        = params.mirostat;
                        const float   mirostat_tau    = params.mirostat_tau;
                        const float   mirostat_eta    = params.mirostat_eta;
                        const bool    penalize_nl     = params.penalize_nl;

                        llama_token id = 0;

                        {
                            auto logits  = llama_get_logits(ctx);
                            auto n_vocab = llama_n_vocab(ctx);

                            // Apply params.logit_bias map
                            for (auto it = params.logit_bias.begin(); it != params.logit_bias.end(); it++) {
                                logits[it->first] += it->second;
                            }

                            std::vector<llama_token_data> candidates;
                            candidates.reserve(n_vocab);
                            for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
                                candidates.emplace_back(llama_token_data{token_id, logits[token_id], 0.0f});
                            }

                            std::vector<llama_token_data> candidates_save = candidates;

                            // The sample_seeds mechanism is a cheat, for the case where we just only
                            // need the next token. We just reroll the selected candidate and
                            // record the selections.
                            // XXX: This really only works if you need one response token!
                            int seeds_remaining = 1;
                            int sample_seeds_idx = -1;
                            if (sample_seeds.size() > 0) {
                                sample_seeds_idx = 0;
                                seeds_remaining = sample_seeds.size();
                            }

                            while (seeds_remaining > 0) {
                                if (sample_seeds_idx >= 0) {
                                    seed = sample_seeds[sample_seeds_idx];
                                    fprintf(stderr, "set seed=%ld\n", seed);
                                    llama_set_rng_seed(ctx, seed);
                                    sample_seeds_idx++;
                                }

                                candidates = candidates_save;

                                llama_token_data_array candidates_p = { candidates.data(), candidates.size(), false };

                                printf("last_n_tokens_size=%d\n", (int) last_n_tokens.size());

                                // Apply penalties
                                float nl_logit = logits[llama_token_nl()];
                                auto last_n_repeat = std::min(std::min((int)last_n_tokens.size(), repeat_last_n), n_ctx);
                                printf("last_n_repeat=%d\n", (int) last_n_repeat);
                                llama_sample_repetition_penalty(ctx, &candidates_p,
                                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                    last_n_repeat, repeat_penalty);
                                llama_sample_frequency_and_presence_penalties(ctx, &candidates_p,
                                    last_n_tokens.data() + last_n_tokens.size() - last_n_repeat,
                                    last_n_repeat, alpha_frequency, alpha_presence);
                                if (!penalize_nl) {
                                    logits[llama_token_nl()] = nl_logit;
                                }

                                if (grammar != NULL) {
                                    llama_sample_grammar(ctx, &candidates_p, grammar);
                                }

                                if (temp <= 0) {
                                    // Greedy sampling
                                    id = llama_sample_token_greedy(ctx, &candidates_p);
                                } else {
                                    if (mirostat == 1) {
                                        static float mirostat_mu = 2.0f * mirostat_tau;
                                        const int mirostat_m = 100;
                                        llama_sample_temperature(ctx, &candidates_p, temp);
                                        id = llama_sample_token_mirostat(ctx, &candidates_p, mirostat_tau, mirostat_eta, mirostat_m, &mirostat_mu);
                                    } else if (mirostat == 2) {
                                        static float mirostat_mu = 2.0f * mirostat_tau;
                                        llama_sample_temperature(ctx, &candidates_p, temp);
                                        id = llama_sample_token_mirostat_v2(ctx, &candidates_p, mirostat_tau, mirostat_eta, &mirostat_mu);
                                    } else {
                                        // Temperature sampling
                                        llama_sample_top_k(ctx, &candidates_p, top_k, 1);
                                        llama_sample_tail_free(ctx, &candidates_p, tfs_z, 1);
                                        llama_sample_typical(ctx, &candidates_p, typical_p, 1);
                                        llama_sample_top_p(ctx, &candidates_p, top_p, 1);
                                        llama_sample_temperature(ctx, &candidates_p, temp);
                                        id = llama_sample_token(ctx, &candidates_p);
                                    }
                                }

                                if (sample_seeds.size() > 0 && id != llama_token_eos()) {
                                    std::vector<llama_token> embd_single_id;
                                    embd_single_id.push_back(id);
                                    j_resps.push_back(make_response(
                                        responses, current_test_nr, total_tests,
                                        test_id, 0.0, seed, embd_single_id));
                                }

                                seeds_remaining -= 1;
                            }

                            if (grammar != NULL) {
                                llama_grammar_accept_token(ctx, grammar, id);
                            }

                            last_n_tokens.erase(last_n_tokens.begin());
                            last_n_tokens.push_back(id);
                        }

                        // add it to the context
                        embd.push_back(id);
                        std::string tok = llama_token_to_str(ctx, id);
                        printf("tok=[%s %d]\n", tok.c_str(), id);

                        if (sample_seeds.size() == 0) {
                            embd_gen.push_back(id);
                        }

                        // decrement remaining sampling budget
                        --n_remain;
                    }

                    // end of text token
                    if (!embd.empty() && embd.back() == llama_token_eos()) {
                        // fprintf(stderr, " [end of text]\n");
                        break;
                    }

                    if (prompt_runner_conf.find("stop_sequences")
                        != prompt_runner_conf.end())
                    {
                        std::string generated = "";
                        for (auto id : embd_gen) {
                            generated += llama_token_to_str(ctx, id);
                        }

                        for (const auto &matched : prompt_runner_conf["stop_sequences"])
                        {
                            if (generated.find(matched) != std::string::npos)
                            {
                                n_remain = 0;
                                break;
                            }
                        }
                    }
                }

                if (sample_seeds.size() == 0) {
                    j_resps.push_back(
                        make_response(
                            responses, current_test_nr, total_tests, test_id,
                            params.temp, seed, embd_gen));
                }

    //            std::string gen_prefix = "[id=" + test_id + ", seed=" + std::to_string(seed) + "]: ";
    //            std::string gen = "";
    //            for (auto id : embd_gen) {
    //                gen += llama_token_to_str(ctx, id);
    //            }
    //            printf("%s %s\n", gen_prefix.c_str(), gen.c_str());
    //            fflush(stdout);
    //
    //            responses.push_back(gen_prefix + gen);
    //
    //            json single_response;
    //            single_response["test_id"] = test_id;
    //            single_response["seed"] = seed;
    //            single_response["response"] = gen;
    //            j_resps.push_back(single_response);


                if (grammar != NULL) {
                    llama_grammar_free(grammar);
                }

                first = false;

                // TODO WEICON:
                // - delete context here
                llama_free(ctx);
                ctx = nullptr;
            }
        }
    }

    // get model file name
    std::string model_file = params.model.c_str();
    model_file = model_file.substr(model_file.find_last_of("/\\") + 1);
    printf("model: %s\n", model_file.c_str());

    // print the collected responses for debugging and to see first results
    for (auto resp : responses) {
        printf("%s\n", resp.c_str());
    }

    // write out the results to a json file
    std::string responses_json_dump = j_resps.dump(2);
    printf("%s\n", responses_json_dump.c_str());
    json results;
    results["model_file"] = model_file;
    results["prompt"] = std::string(params.prompt);
    results["config"] = prompt_runner_conf;
    results["results"] = j_resps;

    std::string out_file_name =
        "result_" + std::to_string(time(NULL)) + "_" + model_file + ".json";
    std::ofstream outf(out_file_name);
    outf << results.dump(2);
    outf.close();

    // cleanup...
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
