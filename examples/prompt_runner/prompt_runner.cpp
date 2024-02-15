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
static bool g_verbose;

static size_t benchmark_start_time;

const char *ws = "\n\r";

void print_kv_cache(const std::string &debug_tok);
void print_kv_cache(const std::string &debug_tok) {
    printf("##### KV CACHE [%s] #####\n", debug_tok.c_str());
    llama_kv_cache_debug_print(*g_ctx);
    printf("#########################\n");
}

std::string concatl(const std::vector<std::string> &l, bool indent = false);

std::string concatl(const std::vector<std::string> &l, bool indent) {
    std::string logstr;
    for (auto logentry : l) {
        if (indent) {
            std::string le = logentry;
            std::replace(le.begin(), le.end(), '\n', ' ');
            logstr += "    " + le + "\n\n";
        } else {
            logstr += logentry;
        }
    }
    return logstr;
}

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

void j2i(json map, const std::string &lbl, int32_t &out);
void j2i(json map, const std::string &lbl, int32_t &out) {
    if (map.find(lbl) != map.end()) {
        out = map.value(lbl, 0);
    }
}

void j2f(json map, const std::string &lbl, float &out);
void j2f(json map, const std::string &lbl, float &out) {
    if (map.find(lbl) != map.end()) {
        out = map.value(lbl, 0.0);
    }
}
void j2s(json map, const std::string &lbl, std::string &out);
void j2s(json map, const std::string &lbl, std::string &out) {
    if (map.find(lbl) != map.end()) {
        out = map.value(lbl, "");
    } else {
        out = "";
    }
}

void json_to_sparams(json sampling, llama_sampling_params &sp);
void json_to_sparams(json sampling, llama_sampling_params &sp) {
    j2i(sampling, "top_k", sp.top_k);
    j2f(sampling, "top_p", sp.top_p);
    j2f(sampling, "tfs_z", sp.tfs_z);
    j2f(sampling, "min_p", sp.min_p);
    j2f(sampling, "typical_p", sp.typical_p);
    j2f(sampling, "temp", sp.temp);
    j2f(sampling, "penalty_present", sp.penalty_present);
    j2f(sampling, "penalty_freq", sp.penalty_freq);
    j2i(sampling, "penalty_last_n", sp.penalty_last_n);
    j2f(sampling, "penalty_repeat", sp.penalty_repeat);
    j2i(sampling, "mirostat", sp.mirostat);
    j2f(sampling, "mirostat_tau", sp.mirostat_tau);
    j2f(sampling, "mirostat_eta", sp.mirostat_eta);
    j2s(sampling, "grammar", sp.grammar);

    if (sampling.find("order") != sampling.end()) {
        std::string order = sampling.value("order", "");
        const auto sampler_names = string_split(order, ';');
        sp.samplers_sequence = sampler_types_from_names(sampler_names);
    }
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
    j_params["grammar"] = sp.grammar;

    return j_params;
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
    int64_t seed;

    PromptRunContext() : cur_test_nr(0), total_tests(0), seed(0) {}
};

json make_token_respose(PromptRunContext &prc, json tokens);

json make_token_respose(PromptRunContext &prc, json tokens) {
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
                           "| id=" + prc.test_id + " #e=" + expected_s + "]: ";

    for (const auto &tok : tokens) {
        char buf[128];
        float prob = tok[1];
        std::string t = tok[0];
        snprintf(buf, 127, "[%s: %1.4f]", t.c_str(), prob);
        info_str += buf;
    }

    printf("[s/t=%5.2fs, eta=%5.1fm, t=%5.1fm] %s\n",
           time_per_test,
           remaining,
           passed_time_mins,
           info_str.c_str());
    fflush(stdout);

    json single_response;
    single_response["test_id"] = prc.test_id;
    single_response["tokens"] = tokens;
    if (!prc.expected.is_null()) {
        single_response["expected"] = prc.expected;
    }
    single_response["timestamp"] = (int)time(NULL);
    single_response["time"] = now_timestr();

    return single_response;
}

json make_response(PromptRunContext &prc,
                   const std::string gen,
                   int gen_tok_cnt,
                   json prompt);

json make_response(PromptRunContext &prc,
                   const std::string gen,
                   int gen_tok_cnt,
                   json prompt) {
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
        ", seed=" + std::to_string(prc.seed) +
        ", #g=" + std::to_string(gen_tok_cnt) + ", #e=" + expected_s + "]:";

    std::string print_gen = gen;
    std::replace(print_gen.begin(), print_gen.end(), '\r', ' ');
    std::replace(print_gen.begin(), print_gen.end(), '\n', '/');
    std::replace(print_gen.begin(), print_gen.end(), '\t', '/');

    printf("[s/t=%5.2fs, eta=%5.1fm, t=%5.1fm] %s %s\n",
           time_per_test,
           remaining,
           passed_time_mins,
           gen_prefix.c_str(),
           print_gen.c_str());
    fflush(stdout);

    json single_response;
    single_response["test_id"] = prc.test_id;
    single_response["seed"] = prc.seed;
    single_response["response"] = gen;
    single_response["expected"] = prc.expected;
    single_response["prompt"] = prompt;
    single_response["prompt_token_count"] = gen_tok_cnt;
    single_response["timestamp"] = (int)time(NULL);
    single_response["time"] = now_timestr();

    return single_response;
}

std::string cleanup_generated_chat_response(std::string gen);

std::string cleanup_unbalanced(const std::string &str,
                               char quot,
                               int &fixed_quote_len);
std::string cleanup_unbalanced(const std::string &str,
                               char quot,
                               int &fixed_quote_len) {
    bool found_open = false;
    size_t idx_last_quot = 0;

    for (size_t i = 0; i < str.size(); i++) {
        if (str[i] == quot) {
            idx_last_quot = i;
            found_open = !found_open;
        }
    }
    //    std::string “”
    if (found_open) {
        std::string ini = str.substr(0, idx_last_quot);
        std::string o = str.substr(idx_last_quot + 1, std::string::npos);
        o = std::regex_replace(
            o, std::regex("^(.*[.?!,])[^.?!,]*$", std::regex::extended), "$1");
        trim_nl(o, " \r\n");
        fixed_quote_len = o.size();
        if (o.size() > 0) {
            auto last_char = o[o.size() - 1];
            if (last_char == ',') {
                o[o.size() - 1] = '.';
                if (quot == '"') {
                    o += "..";
                }
            } else if (quot == '"' && last_char != '?' && last_char != '!' &&
                       last_char != '.') {
                o += "...";
            }
        }
        if (o.size() == 0) {
            o = ini;
            trim_nl(o, " \r\n");
        } else {
            o = ini + quot + o + quot;
        }

        return o;
    }
    return str;
}

void replaceAll(std::string &str,
                const std::string &from,
                const std::string &to);

void replaceAll(std::string &str,
                const std::string &from,
                const std::string &to) {
    if (from.empty()) return;
    size_t start_pos = 0;
    while ((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length();  // In case 'to' contains 'from', like
                                   // replacing 'x' with 'yx'
    }
}

void unifyQuotes(std::string &gen);
void unifyQuotes(std::string &gen) {
    replaceAll(gen, "‚", ",");
    replaceAll(gen, "„", "\"");
    replaceAll(gen, "»", "\"");
    replaceAll(gen, "«", "\"");
    replaceAll(gen, "‘", "'");
    replaceAll(gen, "’", "'");
    replaceAll(gen, "‚", ",");
    replaceAll(gen, "“", "\"");
    replaceAll(gen, "”", "\"");
    replaceAll(gen, "‹", "'");
    replaceAll(gen, "›", "'");
}

bool find_any_of_in_string(const std::string &needles, const std::string &str);
bool find_any_of_in_string(const std::string &needles, const std::string &str) {
    for (size_t i = 0; i < needles.size(); i++) {
        if (str.find(needles[i]) != std::string::npos) {
            return true;
        }
    }
    return false;
}

std::string cleanup_generated_chat_response(std::string gen) {
    gen = std::regex_replace(gen, std::regex("^:", std::regex::extended), "");

    unifyQuotes(gen);

    // Yep, no punctuation means: no full response => garbage
    if (!find_any_of_in_string(".!?*\")}`$", gen)) {
        return "";
    }

    int flen = 0;
    gen = cleanup_unbalanced(gen, '*', flen);
    gen = cleanup_unbalanced(gen, '"', flen);

    // remove newlines and spaces front/end:
    trim_nl(gen, " \r\n");

    // remove newlines in the middle:
    gen =
        std::regex_replace(gen, std::regex("\n\n*", std::regex::extended), " ");

    // Strip trailing incomplete sentences:
    gen = std::regex_replace(
        gen,
        std::regex("^(.*[.!?*\")}`$])[^.!?*\")}`$]*$", std::regex::extended),
        "$1");
    // remove spaces
    trim_nl(gen, " ");

    return gen;
}

struct TextReplacer {
    json prompt_runner_conf;
    json extra_replacements;

    TextReplacer() {}
    TextReplacer(json prconf) : prompt_runner_conf(prconf) {}

    void add_extra(const std::string &key, const std::string &repl) {
        extra_replacements[key] = repl;
    }

    void merge_replacements(json replacements) {
        if (replacements.find("replacements") != replacements.end()) {
            for (const auto &repl : replacements["replacements"]) {
                std::string search = repl[0];
                std::string replacement = repl[1];
                extra_replacements[search] = replacement;
            }
        }
    }

    std::string apply_replacements(std::string prompt) {
        for (const auto &repl : extra_replacements.items()) {
            std::string search = repl.key();
            std::string replacement = repl.value();
            std::string new_t =
                std::regex_replace(prompt, std::regex(search), replacement);
            if (new_t != prompt) {
                prompt = apply_replacements(new_t);
            }
        }

        if (prompt_runner_conf.find("replacements") !=
            prompt_runner_conf.end()) {
            for (const auto &repl : prompt_runner_conf["replacements"]) {
                std::string search = repl[0];
                std::string replacement = repl[1];
                std::string new_t =
                    std::regex_replace(prompt, std::regex(search), replacement);
                if (new_t != prompt) {
                    prompt = apply_replacements(new_t);
                }
            }
        }

        return prompt;
    }

    std::string apply_replacements(json prompt_test, std::string prompt) {
        for (const auto &repl : extra_replacements.items()) {
            std::string search = repl.key();
            std::string replacement = repl.value();
            std::string new_t =
                std::regex_replace(prompt, std::regex(search), replacement);
            if (new_t != prompt) {
                prompt = apply_replacements(prompt_test, new_t);
            }
        }

        if (prompt_runner_conf.find("replacements") !=
            prompt_runner_conf.end()) {
            for (const auto &repl : prompt_runner_conf["replacements"]) {
                std::string search = repl[0];
                std::string replacement = repl[1];
                std::string new_t =
                    std::regex_replace(prompt, std::regex(search), replacement);
                if (new_t != prompt) {
                    prompt = apply_replacements(prompt_test, new_t);
                }
            }
        }

        if (prompt_test.find("replacements") != prompt_test.end()) {
            for (const auto &repl : prompt_test["replacements"]) {
                std::string search = repl[0];
                std::string replacement = repl[1];
                std::string new_t =
                    std::regex_replace(prompt, std::regex(search), replacement);
                if (new_t != prompt) {
                    prompt = apply_replacements(prompt_test, new_t);
                }
            }
        }

        return prompt;
    }
};

struct SampleParamSource {
    llama_sampling_params default_sparams;
    json sampler_lib;

    SampleParamSource(const llama_sampling_params &sparams)
        : default_sparams(sparams) {}
    void set_sampler_lib(json slib) { sampler_lib = slib; }

    bool load_sampler_settings(json in_sampler,
                               llama_sampling_params &sparams) {
        if (in_sampler.is_string()) {
            std::string key = in_sampler;
            if (sampler_lib.find(key) == sampler_lib.end()) {
                printf("ERROR: UNKNOWN SAMPLER KEY: %s\n", key.c_str());
                sparams = default_sparams;
                return false;
            }

            json_to_sparams(sampler_lib[key], sparams);
        } else if (in_sampler.is_object()) {
            json_to_sparams(in_sampler, sparams);
        } else if (in_sampler.is_null()) {
            sparams = default_sparams;
        } else {
            std::string du = in_sampler.dump();
            printf("ERROR: BAD SAMPLER SETTINGS: %s\n", du.c_str());
            return false;
        }

        return true;
    }
};

struct StopSequences {
    json stop_sequences;

    void set_stop_sequences(json ss) { stop_sequences = ss; }

    bool trim_stop_sequence(const std::string &input, std::string &trimmed) {
        for (const auto &matched : stop_sequences) {
            size_t stop_pos = input.find(matched);
            if (stop_pos != std::string::npos) {
                trimmed = input.substr(0, stop_pos);
                return true;
            }
        }

        trimmed = input;

        return false;
    }
};

std::string probs_to_string(
    const std::vector<std::pair<std::string, float>> &probs);

std::string probs_to_string(
    const std::vector<std::pair<std::string, float>> &probs) {
    std::string res;
    for (auto p : probs) {
        char buf[20];
        size_t len = snprintf(buf, 20, "%9.7f", p.second);
        res += std::to_string(p.first.size()) + ":" + p.first + " " +
               std::string(buf, len) + ";";
    }
    return res;
}

struct Completion {
    bool error;
    std::vector<std::pair<std::string, float>> tok_probabilities;
    std::vector<std::string> min_p_tokens;
    int prompt_token_count;
    std::string raw;
    std::string completion;

    Completion() : error(false), prompt_token_count(0) {}

    void set_inference_error() { error = true; }
};

struct CompletionNode {
    std::string name;
    bool skip;
    bool no_commit;
    int index;
    std::string prefix;
    std::string override_text;
    std::string override_postfix;
    std::string override_payload_key;
    bool use_override_text;
    std::string postfix;
    json sampler_settings;
    json payload;
    int gen_count;
    std::string cleanup_quotes;
    std::string cleanup_regex;
    std::string cleanup_regex_repl;
    StopSequences stop_seq;
    int64_t seed;
    float record_min_p_tokens;
    int gen_top_n_tokens;
    json replacements;

    std::vector<std::pair<std::string, float>> result_probs;
    json top_token_results;
    std::string result_string;
    std::string result_raw;
    std::string result_context;
    int prompt_token_count;
    json result_sampler;
    json result_min_p_tokens;

    CompletionNode()
        : skip(false),
          no_commit(false),
          index(0),
          use_override_text(false),
          gen_count(0),
          seed(-1),
          record_min_p_tokens(0.0),
          gen_top_n_tokens(0),
          prompt_token_count(0) {}

    void from_json(
        const json &node,
        const std::vector<std::pair<std::string, json>> &prev_nodes) {
        if (node.find("inherit") != node.end()) {
            std::string inherit_name = node.value("inherit", "");
            for (auto &pn : prev_nodes) {
                if (pn.first == inherit_name) {
                    this->from_json(pn.second, prev_nodes);
                    break;
                }
            }
        }

        name = node.value("name", "");
        skip = node.value("skip", false);

        if (node.find("sampler") != node.end()) {
            sampler_settings = node["sampler"];
        }

        if (node.find("stop_sequences") != node.end()) {
            stop_seq.set_stop_sequences(node["stop_sequences"]);
        }

        if (node.find("prefix") != node.end()) {
            prefix = node.value("prefix", "");
        }

        if (node.find("override_payload_key") != node.end()) {
            override_payload_key = node.value("override_payload_key", "");
            use_override_text = true;
        }

        if (node.find("override") != node.end()) {
            override_text = node.value("override", "");
            use_override_text = true;
        }

        if (node.find("override_postfix") != node.end()) {
            override_postfix = node.value("override_postfix", "");
        }

        if (node.find("postfix") != node.end()) {
            postfix = node.value("postfix", "");
        }

        if (node.find("n_gen") != node.end()) {
            gen_count = node.value("n_gen", 0);
        }

        if (node.find("no_commit") != node.end()) {
            no_commit = node.value("no_commit", false);
        }

        if (node.find("gen_top_n_tokens") != node.end()) {
            gen_top_n_tokens = node.value("gen_top_n_tokens", 0);
        }

        if (node.find("seed") != node.end()) {
            seed = node.value("seed", -1);
        }

        if (node.find("cleanup") != node.end()) {
            json cleanup = node["cleanup"];
            cleanup_quotes = cleanup.value("quotes", "");
            if (cleanup.find("regex") != cleanup.end() &&
                cleanup["regex"].size() > 1) {
                cleanup_regex = cleanup["regex"][0];
                cleanup_regex_repl = cleanup["regex"][1];
            }
        }

        if (node.find("record_min_p_tokens") != node.end()) {
            record_min_p_tokens = node.value("record_min_p_tokens", 0.0);
        }

        if (node.find("payload") != node.end()) {
            payload = node["payload"];
        }

        if (node.find("replacements") != node.end()) {
            replacements = node["replacements"];
        }
    }

    std::string get_override_text(TextReplacer &replacer) {
        std::string ret = get_completion_text(replacer);
        if (override_payload_key.size() > 0) {
            if (payload.find(override_payload_key) != payload.end()) {
                std::string payl = payload.value(override_payload_key, "");
                ret = ret + payl + override_postfix;
            } else {
                ret = ret + override_postfix;
            }
        } else {
            ret = ret + override_text + override_postfix;
        }

        if (postfix.size() > 0) {
            ret = ret + postfix;
        }

        return ret;
    }

    std::string get_completion_text(TextReplacer &replacer) {
        if (!replacements.is_null()) {
            replacer.merge_replacements(replacements);
        }
        return replacer.apply_replacements(prefix);
    }

    void add_top_token_result(
        int top_token_index,
        const std::vector<std::pair<std::string, float>> &probs,
        const std::string &top_result_string,
        const std::vector<std::string> &min_p_tokens) {
        json res;
        res["text"] = top_result_string;
        res["probs"] = probs_to_string(probs);
        res["index"] = top_token_index;
        if (min_p_tokens.size() > 0) {
            json min_p;
            min_p["min_p"] = record_min_p_tokens;
            min_p["res"] = min_p_tokens;
            res["min_p_tokens"] = min_p;
        }
        top_token_results.push_back(res);
    }

    json result_to_json() {
        json res;
        res["text"] = result_string;
        res["raw"] = result_raw;
        res["context"] = result_context;
        res["prompt_token_count"] = prompt_token_count;
        res["probs"] = probs_to_string(result_probs);
        if (!payload.is_null()) {
            res["payload"] = payload;
        }
        if (top_token_results.size() > 0) {
            res["top_token_results"] = top_token_results;
        }
        if (result_min_p_tokens.size() > 0) {
            json min_p;
            min_p["min_p"] = record_min_p_tokens;
            min_p["res"] = result_min_p_tokens;
            res["min_p_tokens"] = min_p;
        }
        // res["sampler"] = result_sampler;
        return res;
    }
};

struct Inference {
    struct Sequence {
        // [p0,p1) describing the position in kv cache
        int p0;
        int p1;
        // kv cache sequence ID
        int seq_id;
        // name of this sequence
        std::string name;
        // name of the previous sequence
        std::string prev_name;
        std::vector<llama_token> tokens;
        // for rewinding
        int recent_add_tokens;
        int base_tokens;
        bool base_committed;
        bool no_bos;

        Sequence()
            : p0(0),
              p1(0),
              seq_id(0),
              recent_add_tokens(0),
              base_tokens(0),
              base_committed(false),
              no_bos(false) {}

        void commit() { recent_add_tokens = 0; }

        void commit_base() { base_committed = true; }

        void add_token(llama_token tok) {
            p1 += 1;
            tokens.push_back(tok);
            recent_add_tokens += 1;

            if (!base_committed) {
                base_tokens += 1;
            }
        }

        void _rewind_by(int n) {
            llama_kv_cache_seq_rm(*g_ctx, seq_id, p1 - n, p1);
            for (int i = 0; i < recent_add_tokens; i++) {
                tokens.pop_back();
                p1 -= 1;
            }
        }

        void remove_n_tokens_after_base(size_t n) {
            int base_len = base_tokens;
            if (!base_committed) base_len = 0;

            int p1_start_seq = p0 + base_len;

            size_t remove_n_toks = n;
            std::vector<llama_token> new_tokens;
            for (size_t i = 0; i < (size_t)base_len && i < tokens.size(); i++) {
                new_tokens.push_back(tokens[i]);
            }
            for (size_t i = ((size_t)base_len) + remove_n_toks;
                 i < tokens.size();
                 i++) {
                new_tokens.push_back(tokens[i]);
            }
            tokens = new_tokens;

            // printf("### KV RM seq=%d, p0=%d to p1=%d\n",
            //        seq_id,
            //        p1_start_seq,
            //        p1_start_seq + n);
            llama_kv_cache_seq_rm(
                *g_ctx, seq_id, p1_start_seq, p1_start_seq + n);
            int p1_rest = p1_start_seq + n;
            // printf("### KV SHIFT seq=%d, p0=%d to p1=%d to p0=%d, p1=%d\n",
            //        seq_id,
            //        p1_rest,
            //        p1,
            //        p1_rest - n,
            //        p1 - n);
            llama_kv_cache_seq_shift(*g_ctx, seq_id, p1_rest, p1, -n);
            p1 -= n;
        }

        void rewind_to_base() {
            int base_len = base_tokens;
            if (!base_committed) base_len = 0;

            int cur_toks = (int)tokens.size();
            if (cur_toks < base_len) {
                return;
            }

            _rewind_by(cur_toks - base_tokens);
        }

        void rewind() {
            _rewind_by(recent_add_tokens);
            recent_add_tokens = 0;
        }

        std::string to_string() {
            std::string str;
            for (auto tok : tokens) {
                str += llama_token_to_piece(*g_ctx, tok);
            }
            return str;
        }

        std::string recent_string() {
            if (recent_add_tokens <= 0) {
                return "";
            }

            if (tokens.size() < (size_t)recent_add_tokens) {
                return "RECENT_TOKENS_BUGBUGBUG";
            }

            std::string str;
            for (size_t i = tokens.size() - recent_add_tokens;
                 i < tokens.size();
                 i++) {
                llama_token tok = tokens[i];
                str += llama_token_to_piece(*g_ctx, tok);
            }

            return str;
        }
    };

    int cur_seq_id;

    int n_batch;
    std::vector<Sequence> sequences;
    std::vector<std::pair<std::string, float>> tok_probs;
    std::vector<std::string> min_p_tok_probs;
    float record_min_p_tokens;
    int sample_from_token_i;
    llama_sampling_params sparams;
    llama_sampling_context *ctx_sampling;

    int last_append_token_count;

    Inference(llama_sampling_params &sp, int nb)
        : cur_seq_id(0),
          n_batch(nb),
          record_min_p_tokens(0.0),
          sample_from_token_i(1),
          sparams(sp),
          last_append_token_count(0) {
        llama_kv_cache_clear(*g_ctx);
        ctx_sampling = llama_sampling_init(sp);
    }

    void reset_seed(int seed) {
        // printf("RESET SEED %d\n", seed);
        llama_set_rng_seed(*g_ctx, seed);
    }

    void clear() {
        sequences.clear();
        llama_kv_cache_clear(*g_ctx);
        llama_sampling_reset(ctx_sampling);
    }

    ~Inference() {
        llama_kv_cache_clear(*g_ctx);
        llama_sampling_free(ctx_sampling);
    }

    int current_max_seq_token_count() {
        int max = 0;
        for (auto seq : sequences) {
            if (seq.p1 > max) {
                max = seq.p1;
            }
        }
        return max;
    }

    std::vector<llama_token> tokenize(const std::string &name,
                                      const std::string &text) {
        bool is_bos = name == "";
        bool add_bos = name == "";
        if (name.size() > 0) {
            Sequence *seq = get_sequence(name);
            is_bos = (!seq->no_bos) && seq->p1 == 0;
            add_bos = (seq->p1 == 0) &&
                      (llama_vocab_type(*g_model) == LLAMA_VOCAB_TYPE_SPM);
        }

        // printf("TOKENIZE add_bos=%d, is_bos=%d [%s]\n",
        //        add_bos,
        //        is_bos,
        //        text.c_str());
        std::vector<llama_token> tokens;
        tokens = ::llama_tokenize(*g_ctx, text.c_str(), add_bos, true, !is_bos);
        return tokens;
    }

    bool load_tokens(const std::string &name,
                     const std::string &text,
                     int seq_id,
                     int p0,
                     std::vector<llama_token> &out_tokens,
                     int &p1) {
        if (text.size() == 0) return true;
        auto tokens = tokenize(name, text);

        for (int i = 0; i < (int)tokens.size(); i += n_batch) {
            int n_eval = (int)tokens.size() - i;
            if (n_eval > n_batch) {
                n_eval = n_batch;
            }

            auto batch = llama_batch_init(tokens.size(), 0, 1);
            for (int j = 0; j < n_eval; j++) {
                int tok_idx = j + i;
                // const std::string piece =
                //     llama_token_to_piece(*g_ctx, tokens[tok_idx]);
                // printf("LDDBATCH %d = %d [%s]\n",
                //        p0 + tok_idx,
                //        tokens[tok_idx],
                //        piece.c_str());
                llama_batch_add(
                    batch, tokens[tok_idx], p0 + tok_idx, {seq_id}, false);
                out_tokens.push_back(tokens[tok_idx]);
                p1 = p0 + tok_idx + 1;
            }

            if (llama_decode(*g_ctx, batch)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                llama_batch_free(batch);
                return false;
            }

            llama_batch_free(batch);
        }

        return true;
    }

    int get_sequence_idx(const std::string &name) {
        for (size_t i = 0; i < sequences.size(); i++) {
            if (sequences[i].name == name) {
                return (int)i;
            }
        }

        return -1;
    }

    Sequence *get_sequence(const std::string &name) {
        int sidx = get_sequence_idx(name);
        Sequence *s = nullptr;
        if (sidx >= 0) {
            s = &sequences[sidx];
        } else {
            sequences.push_back(Sequence());
            s = &sequences[sequences.size() - 1];
            s->seq_id = cur_seq_id++;
            s->name = name;
        }
        return s;
    }

    void commit_base(const std::string &name) {
        int sidx = get_sequence_idx(name);
        if (sidx >= 0) {
            sequences[sidx].commit_base();
        }
    }

    void commit(const std::string &name) {
        int sidx = get_sequence_idx(name);
        if (sidx >= 0) {
            sequences[sidx].commit();
            // printf("COMMIT PROMPT:[%s]\n",
            //        this->get_sequence_text(name).c_str());
        }
    }

    llama_token sample_top_token_i(int sample_idx, int &tok_idx) {
        const int n_vocab = llama_n_vocab(*g_model);

        float *logits = llama_get_logits_ith(*g_ctx, sample_idx);

        std::vector<llama_token_data> cur;
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur.emplace_back(
                llama_token_data{token_id, logits[token_id], 0.0f});
        }

        llama_token_data_array candidates_p = {cur.data(), cur.size(), false};

        if (ctx_sampling->grammar != NULL) {
            llama_sample_grammar(*g_ctx, &candidates_p, ctx_sampling->grammar);
        }

        llama_sample_softmax(nullptr, &candidates_p);

        if (candidates_p.size > 0) {
            if (((size_t)tok_idx) >= candidates_p.size) {
                tok_idx = candidates_p.size - 1;
            }

            return candidates_p.data[tok_idx].id;
        } else {
            return 0;
        }
    }

    float get_token_probabilitiy(llama_token tok, int sample_idx) {
        const int n_vocab = llama_n_vocab(*g_model);

        float *logits = llama_get_logits_ith(*g_ctx, sample_idx);

        std::vector<llama_token_data> cur;
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur.emplace_back(
                llama_token_data{token_id, logits[token_id], 0.0f});
        }

        llama_token_data_array candidates_p = {cur.data(), cur.size(), false};

        llama_sample_softmax(nullptr, &candidates_p);

        for (size_t i = 0; i < candidates_p.size; ++i) {
            if (candidates_p.data[i].id == tok) {
                return candidates_p.data[i].p;
            }
        }

        return -1.0;
    }

    std::vector<std::pair<std::string, float>> get_min_p_tokens(
        int sample_idx, float min_p, size_t min_keep) {
        const int n_vocab = llama_n_vocab(*g_model);

        float *logits = llama_get_logits_ith(*g_ctx, sample_idx);

        std::vector<llama_token_data> cur;
        for (llama_token token_id = 0; token_id < n_vocab; token_id++) {
            cur.emplace_back(
                llama_token_data{token_id, logits[token_id], 0.0f});
        }

        llama_token_data_array candidates_p = {cur.data(), cur.size(), false};

        llama_sample_min_p(nullptr, &candidates_p, min_p, min_keep);

        std::vector<std::pair<std::string, float>> toks;
        for (size_t i = 0; i < candidates_p.size; ++i) {
            llama_token tok = candidates_p.data[i].id;

            const std::string piece = llama_token_to_piece(*g_ctx, tok);
            toks.push_back(
                std::pair<std::string, float>(piece, candidates_p.data[i].p));
        }

        return toks;
    }

    std::vector<std::pair<std::string, float>> get_recent_tok_prob_sequence() {
        return tok_probs;
    }

    std::vector<std::string> get_recent_min_p_tok_probs() {
        return min_p_tok_probs;
    }

    bool complete(const std::string &name,
                  const std::string &text,
                  int n_remain,
                  std::function<bool(int, const std::string &)> check_for_end) {
        tok_probs.clear();
        min_p_tok_probs.clear();
        Sequence *seq = get_sequence(name);
        // print_kv_cache("complete start");
        // printf("complete sequence: seq_id=%d p0=%d, p1=%d\n",
        //        seq->seq_id,
        //        seq->p0,
        //        seq->p1);

        auto tokens = tokenize(name, text);
        auto batch = llama_batch_init(tokens.size(), 0, 1);
        for (size_t i = 0; i < tokens.size(); i++) {
            bool is_last = (i + 1 == tokens.size());
            // const std::string piece = llama_token_to_piece(*g_ctx,
            // tokens[i]); printf("ADDBATCH %d (is_last=%d) = %d [%s]\n",
            //        seq->p1,
            //        is_last,
            //        tokens[i],
            //        piece.c_str());
            llama_batch_add(batch, tokens[i], seq->p1, {seq->seq_id}, is_last);
            seq->add_token(tokens[i]);
            llama_sampling_accept(ctx_sampling, *g_ctx, tokens[i], false);
            //            printf("accept %d\n", tokens[i]);
        }

        int sample_idx = tokens.size() - 1;

        if (llama_decode(*g_ctx, batch)) {
            fprintf(
                stderr, "%s [%s]: failed to eval\n", __func__, text.c_str());
            llama_batch_free(batch);
            return false;
        }

        llama_batch_free(batch);
        if (g_verbose) {
            printf(
                "COMPLETE SEQUENCE[%s: %s] seq_id=%d, p0=%d, p1=%d, size=%d, "
                "n_gen=%d\n",
                seq->name.c_str(),
                text.c_str(),
                seq->seq_id,
                seq->p0,
                seq->p1,
                (int)seq->tokens.size(),
                n_remain);
        }

        int gen_count = 0;
        while (n_remain > 0) {
            llama_token tok = 0;
            if (gen_count == 0 && sample_from_token_i > 0) {
                tok = sample_top_token_i(sample_idx, sample_from_token_i);
            } else {
                tok = llama_sampling_sample(
                    ctx_sampling, *g_ctx, nullptr, sample_idx);
            }

            gen_count += 1;

            float tok_p = get_token_probabilitiy(tok, sample_idx);
            const std::string piece = llama_token_to_piece(*g_ctx, tok);

            if (record_min_p_tokens > 0.00001) {
                auto min_p_toks =
                    get_min_p_tokens(sample_idx, record_min_p_tokens, 1);
                min_p_tok_probs.push_back(probs_to_string(min_p_toks));
            }

            tok_probs.push_back(std::pair<std::string, float>(piece, tok_p));

            llama_sampling_accept(ctx_sampling, *g_ctx, tok, true);

            if (tok == llama_token_eos(*g_model)) {
                return true;
            }

            seq->add_token(tok);

            std::string seq_str = seq->recent_string();
            // printf("SEQ %d %s\n", seq->recent_add_tokens, seq_str.c_str());
            if (check_for_end(gen_count, seq_str)) {
                n_remain = 0;
            } else {
                n_remain -= 1;
            }

            auto one_batch = llama_batch_init(tokens.size(), 0, 1);

            // const std::string piece_add = llama_token_to_piece(*g_ctx, tok);
            // printf("ADDBATCH %d (is_last=yes) = %d [%s]\n", seq->p1 - 1,
            // tok, piece_add.c_str());
            llama_batch_add(one_batch, tok, seq->p1 - 1, {seq->seq_id}, true);
            if (llama_decode(*g_ctx, one_batch)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                llama_batch_free(one_batch);
                return false;
            }

            // print_kv_cache(
            //     "complete step sample_idx=" + std::to_string(sample_idx) +
            //     "; token=[" + piece + "]");

            sample_idx = 0;
            llama_batch_free(one_batch);
        }

        return true;
    }

    Completion complete_and_rewind(const std::string &sequence_name,
                                   const std::string prefix,
                                   int token_cnt,
                                   StopSequences &stop_seq) {
        if (!complete(sequence_name,
                      prefix,
                      token_cnt,
                      [&](int, const std::string &comp) {
                          std::string tr;
                          return stop_seq.trim_stop_sequence(
                              comp.substr(prefix.size()), tr);
                      })) {
            Completion c;
            c.set_inference_error();
            return c;
            //            end_reason = "inference error";
            //            fprintf(stderr, "Inference error!\n");
            //            fflush(stderr);
            //            return false;
            //            break;
        }

        Completion c;
        c.tok_probabilities = get_recent_tok_prob_sequence();
        c.min_p_tokens = get_recent_min_p_tok_probs();
        c.prompt_token_count = get_sequence_token_count(sequence_name);
        c.raw = get_recently_added_tokens_str(sequence_name);
        c.completion = c.raw.substr(prefix.size());

        rewind(sequence_name);

        return c;
    }

    Completion complete_node(const std::string &sequence_name,
                             CompletionNode &node,
                             SampleParamSource &sampler_source,
                             TextReplacer &replacer) {
        Completion c;

        sample_from_token_i = 1;

        if (node.gen_count > 0) {
            if (node.gen_top_n_tokens > 0) {
                sample_from_token_i = node.gen_top_n_tokens;
            }

            std::string append_text;

            while (sample_from_token_i > 0) {
                sample_from_token_i -= 1;

                llama_sampling_params test_sparams;
                if (!sampler_source.load_sampler_settings(node.sampler_settings,
                                                          test_sparams)) {
                    c.set_inference_error();
                    return c;
                }

                llama_sampling_free(ctx_sampling);
                ctx_sampling = llama_sampling_init(test_sparams);

                if (node.seed >= 0) {
                    reset_seed(node.seed);
                }

                if (node.record_min_p_tokens > 0.0001) {
                    record_min_p_tokens = node.record_min_p_tokens;
                }

                std::string compl_text = node.get_completion_text(replacer);

                c = complete_and_rewind(
                    sequence_name, compl_text, node.gen_count, node.stop_seq);

                record_min_p_tokens = 0.0;

                if (c.error) {
                    return c;
                }

                append_text = c.raw;

                if (node.cleanup_quotes.size() > 0) {
                    for (size_t i = 0; i < node.cleanup_quotes.size(); i++) {
                        int fixed_quote_len = -1;
                        append_text = cleanup_unbalanced(append_text,
                                                         node.cleanup_quotes[i],
                                                         fixed_quote_len);
                    }
                }

                if (node.cleanup_regex.size() > 0) {
                    append_text =
                        std::regex_replace(append_text,
                                           std::regex(node.cleanup_regex),
                                           node.cleanup_regex_repl);
                }

                if (node.postfix.size() > 0) {
                    append_text = append_text + node.postfix;
                }

                if (node.gen_top_n_tokens > 0) {
                    node.add_top_token_result(sample_from_token_i,
                                              c.tok_probabilities,
                                              append_text,
                                              c.min_p_tokens);
                }

                node.result_raw = c.raw;
                node.result_string = append_text;
                node.result_sampler = sparams_to_json(test_sparams);
            }

            node.prompt_token_count = c.prompt_token_count;
            node.result_probs = c.tok_probabilities;
            node.result_min_p_tokens = c.min_p_tokens;

            if (node.use_override_text) {
                append_text = node.get_override_text(replacer);
            }

            if (!node.no_commit) {
                node.result_context = append_text;
                if (!append(sequence_name, append_text, true)) {
                    c.set_inference_error();
                    return c;
                }
            }

        } else {
            std::string append_text = node.prefix;

            if (node.postfix.size() > 0) {
                append_text = append_text + node.postfix;
            }

            if (node.use_override_text) {
                append_text = node.get_override_text(replacer);
            }

            node.result_raw = append_text;
            node.result_context = append_text;
            if (!node.no_commit) {
                if (!append(sequence_name, append_text)) {
                    c.set_inference_error();
                    return c;
                }
            }

            node.result_string = get_recently_added_tokens_str(sequence_name);
            node.prompt_token_count = get_sequence_token_count(sequence_name);
        }

        if (!node.no_commit) {
            commit(sequence_name);
        }
        return c;
    }

    std::string get_sequence_text_no_prev(const std::string &name) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return "";
        }
        Sequence *seq = &sequences[sidx];

        return seq->to_string();
    }

    int get_sequence_token_count(const std::string &name) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return 0;
        }
        Sequence *seq = &sequences[sidx];

        int prev_add = 0;
        std::string prev_name = sequences[sidx].prev_name;
        if (prev_name.size() > 0) {
            prev_add = get_sequence_token_count(prev_name);
        }

        return prev_add + (int)seq->tokens.size();
    }

    int get_sequence_token_count_no_prev(const std::string &name) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return 0;
        }
        Sequence *seq = &sequences[sidx];

        return (int)seq->tokens.size();
    }

    std::string get_sequence_text(const std::string &name) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return "";
        }
        Sequence *seq = &sequences[sidx];

        std::string add;
        std::string prev_name = sequences[sidx].prev_name;
        if (prev_name.size() > 0) {
            add = get_sequence_text(prev_name);
        }

        return add + seq->to_string();
    }

    std::string get_recently_added_tokens_str(const std::string &name) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return "";
        }
        Sequence *seq = &sequences[sidx];

        return seq->recent_string();
    }

    bool rewind(const std::string &name) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return false;
        }
        Sequence *seq = &sequences[sidx];

        // print_kv_cache("PRE REWIND");
        if (seq->recent_add_tokens > 0) {
            seq->rewind();
        }
        // print_kv_cache("POST REWIND");

        return true;
    }

    bool remove_and_shift_seq(const std::string &name, int delete_token_count) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return false;
        }
        Sequence *seq = &sequences[sidx];

        seq->remove_n_tokens_after_base((size_t)delete_token_count);

        return true;
    }

    bool rewind_to_base(const std::string &name) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return false;
        }
        Sequence *seq = &sequences[sidx];

        // llama_kv_cache_debug_print(*g_ctx, "bef_rewind");
        seq->rewind_to_base();
        // llama_kv_cache_debug_print(*g_ctx, "aft_rewind");

        return true;
    }

    bool append(const std::string &name,
                const std::string &text,
                bool no_bos = false) {
        Sequence *s = get_sequence(name);
        s->no_bos = no_bos;
        // d// printf("SEQUENCE [%s] seq_id=%d\n", name.c_str(), s->seq_id);

        std::vector<llama_token> tokens;
        int new_p1 = 0;
        bool ok = load_tokens(name, text, s->seq_id, s->p1, tokens, new_p1);
        last_append_token_count = 0;
        for (auto tok : tokens) {
            s->add_token(tok);
            last_append_token_count++;
        }

        // print_kv_cache("append end");
        return ok;
    }

    int get_last_append_token_count() { return last_append_token_count; }

    void reset_sampler(const std::string &name) {
        int sidx = get_sequence_idx(name);
        if (sidx < 0) {
            return;
        }

        std::string prev_name = sequences[sidx].prev_name;
        if (prev_name.size() > 0) {
            llama_sampling_reset(ctx_sampling);
            reset_sampler(prev_name);
        } else {
            llama_sampling_reset(ctx_sampling);
        }

        for (auto tok : sequences[sidx].tokens) {
            llama_sampling_accept(ctx_sampling, *g_ctx, tok, false);
            //            printf("accept[%s] %d\n", name.c_str(), tok);
        }
    }
};

struct CompletionScript {
    std::vector<CompletionNode> nodes;
    std::vector<std::pair<std::string, json>> node_configs;

    void from_json(const json &prompt_test) {
        if (prompt_test.find("script") != prompt_test.end()) {
            int index = 0;
            for (auto &node : prompt_test["script"]) {
                std::string name = node.value("name", "");
                node_configs.push_back(
                    std::pair<std::string, json>(name, node));
            }

            for (auto node : prompt_test["script"]) {
                CompletionNode cn;
                cn.index = index;
                cn.from_json(node, node_configs);
                nodes.push_back(cn);
                index += 1;
            }
        }
    }
};

struct Character {
    TextReplacer replacer;
    std::string name;
    std::string prompt;
    std::string log_fmt;
    std::string log_first_fmt;
    std::string next_fmt;
    std::string first_message;
    int n_gen_tokens;

    Character() : n_gen_tokens(0) {}

    void load(const TextReplacer &repl,
              json def,
              const std::string other_name,
              int n_predict_default) {
        replacer = repl;

        name = def.value("name", "Anon");

        replacer.merge_replacements(def);
        replacer.add_extra("\\{bot\\}", name);
        replacer.add_extra("\\{BOT\\}", name);
        replacer.add_extra("\\{char\\}", name);
        replacer.add_extra("\\{CHAR\\}", name);
        replacer.add_extra("\\{\\{char\\}\\}", name);
        replacer.add_extra("\\{\\{CHAR\\}\\}", name);
        replacer.add_extra("\\{user\\}", other_name);
        replacer.add_extra("\\{USER\\}", other_name);
        replacer.add_extra("\\{\\{user\\}\\}", other_name);
        replacer.add_extra("\\{\\{USER\\}\\}", other_name);
        replacer.add_extra("<BOT>", name);
        replacer.add_extra("<bot>", name);
        replacer.add_extra("<CHAR>", name);
        replacer.add_extra("<char>", name);
        replacer.add_extra("<USER>", other_name);
        replacer.add_extra("<user>", other_name);

        prompt = replacer.apply_replacements(def.value("prompt", ""));
        log_fmt = def.value("log_fmt", "<BOT>: <RESPONSE>\n");
        log_first_fmt = def.value("log_first_fmt", log_fmt);
        next_fmt = replacer.apply_replacements(def.value("next_fmt", "<BOT>:"));
        first_message = replacer.apply_replacements(def.value("first_mes", ""));
        n_gen_tokens = def.value("n_gen", n_predict_default);
    }

    std::string format_first(const std::string &response) {
        replacer.add_extra("<RESPONSE>", response);
        return replacer.apply_replacements(log_first_fmt);
    }

    std::string format_for_log(const std::string &response) {
        replacer.add_extra("<RESPONSE>", response);
        return replacer.apply_replacements(log_fmt);
    }

    std::string format_for_next() { return next_fmt; }
};

struct AppendChatResult {
    bool do_reroll;
    bool reroll_broken_quotes;
    std::string log_entry;

    AppendChatResult() : do_reroll(false), reroll_broken_quotes(false) {}

    void broken_quotes() {
        printf("REROLL (BROKEN QUOTES)\n");
        do_reroll = true;
        reroll_broken_quotes = true;
    }
};

struct Conversation {
    struct ChatlogEntry {
        int prompt_token_count;
        int reroll_count;
        int token_count;
        int total_token_count;
        bool shifted;
        bool is_char;
        std::string text;
        std::string raw;
        std::vector<std::pair<std::string, float>> probs;

        ChatlogEntry()
            : prompt_token_count(0),
              reroll_count(0),
              token_count(0),
              total_token_count(0),
              shifted(false),
              is_char(false) {}
        ChatlogEntry(const std::string &txt, bool ichar) {
            prompt_token_count = 0;
            reroll_count = 0;
            token_count = 0;
            total_token_count = 0;
            is_char = ichar;
            shifted = false;
            text = txt;
            raw = txt;
        }
    };

    struct RerollEntry {
        int index;
        int reroll_count;
        std::string reason;
        std::string raw;
        RerollEntry() : index(0), reroll_count(0) {}
    };

    TextReplacer replacer;
    StopSequences stop_seq;

    json char_cfg;
    json user_cfg;

    Character c_user;
    Character c_char;

    json test_replacements;

    int turns;
    int prompt_limit_tok_count;
    int shift_context_chatlog_entries;
    int init_user_prompt_tokens;
    int init_char_prompt_tokens;

    int reroll_count;
    int rerolled_broken_quotes;
    int rerolled_empty_replies;

    std::vector<ChatlogEntry> chatlog;
    std::vector<RerollEntry> rerolls;

    Conversation(TextReplacer repl, StopSequences ss, json test_replacement)
        : replacer(repl),
          stop_seq(ss),
          test_replacements(test_replacement),
          turns(0),
          prompt_limit_tok_count(0),
          shift_context_chatlog_entries(0),
          init_user_prompt_tokens(0),
          init_char_prompt_tokens(0),
          reroll_count(0),
          rerolled_broken_quotes(0),
          rerolled_empty_replies(0) {}

    bool load_config(json chat, int n_predict_default) {
        if (chat.contains("user")) {
            user_cfg = chat["user"];
        } else {
            fprintf(stderr, "BAD CHAT, NO \"user\" KEY!\n");
            return false;
        }

        if (chat.contains("char")) {
            char_cfg = chat["char"];
        } else {
            fprintf(stderr, "BAD CHAT, NO \"user\" KEY!\n");
            return false;
        }

        replacer.merge_replacements(test_replacements);

        c_char.load(replacer,
                    char_cfg,
                    user_cfg.value("name", "Ayumi"),
                    n_predict_default);
        c_user.load(replacer,
                    user_cfg,
                    char_cfg.value("name", "Anon"),
                    n_predict_default);

        turns = chat.value("turns", 100);
        prompt_limit_tok_count = chat.value("context_max_limit", 3900);
        shift_context_chatlog_entries =
            chat.value("shift_chatlog_entries_at_context_max_limit", 0);

        if (c_char.first_message != "") {
            std::string entry = c_char.format_for_log(c_char.first_message);
            chatlog.push_back(ChatlogEntry(entry, true));
        }

        return true;
    }

    int prompt_limit() { return prompt_limit_tok_count; }
    int chat_turns() { return turns; }
    int shifted_chatlog_count() {
        int count = 0;
        for (auto l : chatlog) {
            if (l.shifted) count += 1;
        }
        return count;
    }
    int get_shift_chatlog_entries(int prompt_token_count) {
        if (shift_context_chatlog_entries <= 0) {
            return -1;
        }

        if (prompt_token_count > prompt_limit()) {
            return shift_context_chatlog_entries;
        }

        return 0;
    }

    std::string next_completion_fmt(bool is_user) {
        return is_user ? c_user.format_for_next() : c_char.format_for_next();
    }

    int next_completion_tok_count(bool is_user) {
        return is_user ? c_user.n_gen_tokens : c_char.n_gen_tokens;
    }

    std::string append_raw_chat_response(
        int turn,
        bool is_user,
        const std::string &resp,
        const std::string &raw,
        int prompt_token_count,
        std::vector<std::pair<std::string, float>> &probs) {
        int fixed_quote_len = -1;
        cleanup_unbalanced(resp, '"', fixed_quote_len);
        if (fixed_quote_len > -1 && fixed_quote_len < 10) {
            printf("REROLL\n");
            reroll_count++;
            rerolled_broken_quotes++;
            append_reroll(turn, reroll_count, "unbalanced_quote", raw);
            return "";
        }

        fixed_quote_len = -1;
        cleanup_unbalanced(resp, '*', fixed_quote_len);
        if (fixed_quote_len > -1 && fixed_quote_len < 10) {
            printf("REROLL\n");
            reroll_count++;
            rerolled_broken_quotes++;
            append_reroll(turn, reroll_count, "unbalanced_action_quote", raw);
            return "";
        }

        std::string cleaned;
        stop_seq.trim_stop_sequence(resp, cleaned);
        cleaned = cleanup_generated_chat_response(cleaned);

        trim_nl(cleaned, " \r\n");

        if (cleaned.size() == 0) {
            rerolled_empty_replies++;
            reroll_count++;
            append_reroll(turn, reroll_count, "incomplete_sequence", raw);
            return "";
        }

        Character *c = is_user ? &c_user : &c_char;
        std::string entry = c->format_for_log(cleaned);

        ChatlogEntry ce;
        ce.prompt_token_count = prompt_token_count;
        ce.text = entry;
        ce.raw = raw;
        ce.reroll_count = reroll_count;
        ce.is_char = !is_user;
        ce.probs = probs;
        chatlog.push_back(ce);

        return entry;
    }

    void append_reroll(int conversation_index,
                       int reroll_count,
                       const std::string &reason,
                       const std::string &raw) {
        RerollEntry re;
        re.index = conversation_index;
        re.reroll_count = reroll_count;
        re.reason = reason;
        re.raw = raw;
        rerolls.push_back(re);
    }

    int get_total_token_count() {
        if (chatlog.size() > 0) {
            return chatlog[chatlog.size() - 1].total_token_count;
        }
        return 0;
    }

    void set_init_seq_lengths(int char_n_tok, int user_n_tok) {
        init_char_prompt_tokens = char_n_tok;
        init_user_prompt_tokens = user_n_tok;
    }

    void set_last_chat_response_len(int n_tok) {
        if (chatlog.size() > 0) {
            chatlog[chatlog.size() - 1].token_count = n_tok;

            int user_token_count = init_user_prompt_tokens;
            int char_token_count = init_char_prompt_tokens;
            int total_token_count = 0;
            for (auto &l : chatlog) {
                total_token_count += l.token_count;

                if (l.is_char) {
                    char_token_count =
                        init_char_prompt_tokens + total_token_count;
                    l.total_token_count = char_token_count;
                } else {
                    user_token_count =
                        init_user_prompt_tokens + total_token_count;
                    l.total_token_count = user_token_count;
                }

                // printf("TOK iut=%d ict=%d char=%d tc=%d tot=%d\n",
                //        init_user_prompt_tokens,
                //        init_char_prompt_tokens,
                //        l.is_char,
                //        l.token_count,
                //        l.total_token_count);
            }
        }
    }

    std::string concatl(const std::vector<ChatlogEntry> &l, bool indent) {
        std::string logstr;
        for (auto logentry : l) {
            if (indent) {
                std::string le = logentry.text;
                le = "(" + std::to_string(logentry.total_token_count) + ") " +
                     le;
                if (logentry.shifted) {
                    le = "*S* " + le;
                }
                std::replace(le.begin(), le.end(), '\n', ' ');
                logstr += "    " + le + "\n\n";
            } else {
                logstr += logentry.text;
            }
        }
        return logstr;
    }

    std::string user_prompt() { return c_user.prompt; }
    std::string char_prompt() { return c_char.prompt; }

    std::string chatlog_text() { return concatl(chatlog, false); }

    std::string chatlog_text_ext() { return concatl(chatlog, true); }

    int hide_n_chatlog_entries(int n) {
        int removed_tokens = 0;
        bool hid_one = true;
        while (hid_one && n > 0) {
            hid_one = false;
            for (int i = 0; i < (int)chatlog.size(); i++) {
                if (!chatlog[i].shifted) {
                    n--;
                    hid_one = true;
                    chatlog[i].shifted = true;
                    removed_tokens += chatlog[i].token_count;
                    break;
                }
            }
        }
        return removed_tokens;
    }

    json raw_chatlog_as_json() {
        json log;
        for (auto l : chatlog) {
            log.push_back(l.raw);
        }
        return log;
    }

    json text_chatlog_as_json() {
        json log;
        for (auto l : chatlog) {
            log.push_back(l.text);
        }
        return log;
    }

    json rerolls_as_json() {
        json rolls;
        for (auto re : rerolls) {
            json roll;
            roll["raw"] = re.raw;
            roll["index"] = re.index;
            roll["count"] = re.reroll_count;
            roll["reason"] = re.reason;
            rolls.push_back(roll);
        }
        return rolls;
    }

    json chatlog_as_json() {
        json log;
        for (auto l : chatlog) {
            json entry;
            entry["prompt_token_count"] = l.prompt_token_count;
            entry["total_token_count"] = l.total_token_count;
            entry["reroll_count"] = l.reroll_count;
            entry["is_char"] = l.is_char;
            entry["text"] = l.text;
            entry["raw"] = l.raw;
            entry["shifted"] = l.shifted;
            entry["token_count"] = l.token_count;
            entry["probs"] = probs_to_string(l.probs);
            log.push_back(entry);
        }
        return log;
    }

    void log_chatlog_to_file(int seed,
                             const std::string &model_file,
                             json prompt_test,
                             llama_sampling_params &sp) {
        std::string test_id = prompt_test.value("id", "unknown_test_id");
        json sinfo = sparams_to_json(sp);

        std::string out_file_name = "chatlog_" + model_file + "_" + test_id +
                                    "_" + std::to_string(time(NULL)) + ".md";
        std::ofstream outf(out_file_name);
        outf << "# Chatlog for Test ID " << test_id << ", seed " << seed
             << "\n\n";
        outf << "Model Filename: " << model_file << "\n\n";
        outf << "## Sampling Parameters\n\n```json\n";
        outf << sinfo.dump(2, ' ', false, json::error_handler_t::replace);
        outf << "\n```\n\n";
        outf << "## Character Prompt '" << c_char.name << "'\n\n";
        outf << "```\n";
        outf << c_char.prompt;
        outf << "\n```\n\n";
        outf << "## User Prompt '" << c_user.name << "'\n\n";
        outf << "```\n";
        outf << c_user.prompt;
        outf << "\n```\n\n";
        outf << "## Chatlog\n\n";
        outf << chatlog_text_ext();
        outf << "\n";
        outf.close();
        printf("WROTE FILE %s\n", out_file_name.c_str());
        fflush(stdout);

        json chatlog;
        chatlog["model_file"] = model_file;
        chatlog["sampling"] = sinfo;
        chatlog["chatlog"] = chatlog_as_json();
        chatlog["rerolls"] = rerolls_as_json();
        chatlog["timestamp"] = std::to_string(time(NULL));
        chatlog["time"] = now_timestr();
        std::string jout_file_name = "chatlog_" + model_file + "_" + test_id +
                                     "_" + std::to_string(time(NULL)) + ".json";
        std::ofstream joutf(jout_file_name);
        joutf << chatlog.dump(2, ' ', false, json::error_handler_t::replace);
        joutf.close();
        printf("WROTE FILE %s\n", jout_file_name.c_str());
        fflush(stdout);
    }
};

void record_token_info(json &j_tok_resps,
                       PromptRunContext &prc,
                       struct llama_sampling_context *ctx_sampling,
                       int sample_idx);

void record_token_info(json &j_tok_resps,
                       PromptRunContext &prc,
                       struct llama_sampling_context *ctx_sampling,
                       int sample_idx) {
    const int n_vocab = llama_n_vocab(*g_model);

    float *logits = llama_get_logits_ith(*g_ctx, sample_idx);

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
        j_tok_resps.push_back(make_token_respose(prc, tokens));
    }
}

bool chatlog_has_repetitions(const std::vector<std::string> &chatlog,
                             std::string &piece,
                             std::string &prevlog);

bool chatlog_has_repetitions(const std::vector<std::string> &chatlog,
                             std::string &piece,
                             std::string &prevlog) {
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

std::string process_text_for_console(std::string print_gen);

std::string process_text_for_console(std::string print_gen) {
    std::replace(print_gen.begin(), print_gen.end(), '\r', ' ');
    std::replace(print_gen.begin(), print_gen.end(), '\n', '/');
    std::replace(print_gen.begin(), print_gen.end(), '\t', '/');
    return print_gen;
}

void print_status(int prompt_max_len,
                  int i,
                  const std::string &log_entry,
                  PromptRunContext &prun_ctx,
                  Conversation &conversation,
                  Inference &infer);
void print_status(int prompt_max_len,
                  int i,
                  const std::string &log_entry,
                  PromptRunContext &prun_ctx,
                  Conversation &conversation,
                  Inference &infer) {
    float chat_fraction = ((float)i) / ((float)conversation.chat_turns());
    int passed_time = time(NULL) - benchmark_start_time;
    float passed_tests_f =
        (((float)(prun_ctx.cur_test_nr - 1)) + chat_fraction);
    float time_per_test =
        passed_tests_f > 0.01 ? ((float)passed_time) / passed_tests_f : 0.0;
    float remaining =
        time_per_test * (((float)prun_ctx.total_tests) - passed_tests_f);
    remaining /= 60.0;

    float passed_time_mins = ((float)passed_time) / 60.0;

    std::string print_gen = process_text_for_console(log_entry);
    printf(
        "[test_id=%s, eta=%5.1fm, t=%5.1fm, seed=%ld, test_nr=%d, "
        "total_tests=%d, turn=(%d)%d/%d, plen(%d)=%d/%d, pusr=%d, "
        "pchr=%d]: %s\n",
        prun_ctx.test_id.c_str(),
        remaining,
        passed_time_mins,
        prun_ctx.seed,
        prun_ctx.cur_test_nr,
        prun_ctx.total_tests,
        conversation.shifted_chatlog_count(),
        i + 1,
        conversation.chat_turns(),
        conversation.get_total_token_count(),
        infer.current_max_seq_token_count(),
        prompt_max_len,
        infer.get_sequence_token_count("user"),
        infer.get_sequence_token_count("char"),
        print_gen.c_str());
    fflush(stdout);
}

void print_script_status(int i,
                         int node_count,
                         const std::string &log_entry,
                         PromptRunContext &prun_ctx,
                         Inference &infer);

void print_script_status(int i,
                         int node_count,
                         const std::string &log_entry,
                         PromptRunContext &prun_ctx,
                         Inference &infer) {
    float script_fraction = ((float)i) / ((float)node_count);
    int passed_time = time(NULL) - benchmark_start_time;
    float passed_tests_f =
        (((float)(prun_ctx.cur_test_nr - 1)) + script_fraction);
    float time_per_test =
        passed_tests_f > 0.01 ? ((float)passed_time) / passed_tests_f : 0.0;
    float remaining =
        time_per_test * (((float)prun_ctx.total_tests) - passed_tests_f);
    remaining /= 60.0;

    float passed_time_mins = ((float)passed_time) / 60.0;

    std::string print_gen = process_text_for_console(log_entry);
    printf(
        "[test_id=%s, eta=%5.1fm, t=%5.1fm, seed=%ld, test_nr=%d, "
        "total_tests=%d, turn=%d/%d, plen=%d"
        "]: %s\n",
        prun_ctx.test_id.c_str(),
        remaining,
        passed_time_mins,
        prun_ctx.seed,
        prun_ctx.cur_test_nr,
        prun_ctx.total_tests,
        i + 1,
        node_count,
        infer.current_max_seq_token_count(),
        print_gen.c_str());
    fflush(stdout);
}

// "chat": {
//   "user": {
//       "prompt": "<PROMPT2><CHATLOG>Loki: ",
//       "n_gen": 70,
//       "log_fmt": "Loki: <RESPONSE>\n"
//   },
//   "char": {
//       "prompt": "<PROMPT><CHATLOG>Aria: ",
//       "n_gen": 70,
//       "log_fmt": "Aria: <RESPONSE>\n"
//   },
//   "char_log_init": "Arias character and scenario
//   described here.", "turns": 50
// },
bool chatlog_generator(PromptRunContext &prun_ctx,
                       gpt_params &params,
                       Inference &infer,
                       TextReplacer &replacer,
                       json prompt_runner_conf,
                       json prompt_test,
                       json &j_resps);

bool chatlog_generator(PromptRunContext &prun_ctx,
                       gpt_params &params,
                       Inference &infer,
                       TextReplacer &replacer,
                       json prompt_runner_conf,
                       json prompt_test,
                       json &j_resps) {
    StopSequences stop_seq;
    if (prompt_runner_conf.find("stop_sequences") != prompt_runner_conf.end()) {
        stop_seq.set_stop_sequences(prompt_runner_conf["stop_sequences"]);
    }

    Conversation conversation(replacer, stop_seq, prompt_test);
    conversation.load_config(prompt_test["chat"], params.n_predict);

    if (!infer.append("char", conversation.char_prompt())) {
        fprintf(stderr, "Couldn't init append char prompt\n");
        fflush(stderr);
        return false;
    }
    if (!infer.append("user", conversation.user_prompt())) {
        fprintf(stderr, "Couldn't init append user prompt\n");
        fflush(stderr);
        return false;
    }

    infer.commit_base("char");
    infer.commit_base("user");

    // Set the initial length of the prompts:
    conversation.set_init_seq_lengths(infer.get_sequence_token_count("char"),
                                      infer.get_sequence_token_count("user"));

    if (!infer.append("char", conversation.chatlog_text())) {
        fprintf(stderr, "Couldn't append chatlog\n");
        fflush(stderr);
        return false;
    }
    infer.commit("char");

    conversation.set_last_chat_response_len(
        infer.get_last_append_token_count());

    if (!infer.append("user", conversation.chatlog_text())) {
        fprintf(stderr, "Couldn't append chatlog\n");
        fflush(stderr);
        return false;
    }
    infer.commit("user");

    bool is_user = true;
    std::string end_reason;

    int prompt_max_len = conversation.prompt_limit();

    for (int i = 0; i < conversation.chat_turns(); i++) {
        int user_max_tokens = infer.get_sequence_token_count("user");
        int char_max_tokens = infer.get_sequence_token_count("char");

        int max_tokens = user_max_tokens > char_max_tokens ? user_max_tokens
                                                           : char_max_tokens;
        int shift_chat_log_count =
            conversation.get_shift_chatlog_entries(max_tokens);
        if (shift_chat_log_count < 0) {
            if (max_tokens > prompt_max_len) {
                end_reason = "context limit reached (" +
                             std::to_string(prompt_max_len) + ")";
                break;
            }

        } else if (shift_chat_log_count > 0) {
            int n_tok =
                conversation.hide_n_chatlog_entries(shift_chat_log_count);

            // printf("####KVCACHE BEFORE####\n");
            // llama_kv_cache_debug_print(*g_ctx);

            infer.remove_and_shift_seq("user", n_tok);
            infer.remove_and_shift_seq("char", n_tok);

            // printf("####KVCACHE AFTER####\n");
            // llama_kv_cache_debug_print(*g_ctx);

            // printf("SHIFT by %d tokens! NEW
            // CHATLOG[[[[\n%s\n]]]]\n",
            //    n_tok,
            //    conversation.chatlog_text_ext().c_str());
        }

        if (conversation.reroll_count > 20) {
            end_reason = "reroll limit reached (10)";
            break;
        }

        std::string sequence_name = is_user ? "user" : "char";

        // printf("SEQ[log %d]=%s\n",
        //        infer.current_max_seq_token_count(),
        //        infer.get_sequence_text(sequence_name).c_str());
        // fflush(stdout);

        // TODO: Fetch grammar here!
        infer.reset_seed(prun_ctx.seed + i + conversation.reroll_count);
        infer.reset_sampler(sequence_name);

        // TODO: Replace the following completion and prompt preparation if
        //       there is a scripted response!
        std::string completion_start =
            conversation.next_completion_fmt(is_user);
        int completion_token_cnt =
            conversation.next_completion_tok_count(is_user);

        std::string user_first_msg = "";
        if (conversation.c_user.first_message != "" && i < 2 && is_user) {
            user_first_msg = conversation.c_user.first_message;
            completion_start = completion_start + " " + user_first_msg;
        }

        Completion com = infer.complete_and_rewind(
            sequence_name, completion_start, completion_token_cnt, stop_seq);
        if (com.error) {
            // end_reason = "inference error";
            fprintf(stderr, "Inference error!\n");
            fflush(stderr);
            return false;
        }

        std::string completion = com.completion;
        if (user_first_msg != "") {
            completion = user_first_msg + completion;
        }

        std::string log_entry =
            conversation.append_raw_chat_response(i,
                                                  is_user,
                                                  completion,
                                                  com.raw,
                                                  com.prompt_token_count,
                                                  com.tok_probabilities);
        if (log_entry == "") {
            continue;
        }

        if (!infer.append("user", log_entry)) {
            fprintf(stderr, "Couldn't append chatlog\n");
            fflush(stderr);
            return false;
        }
        infer.commit("user");

        conversation.set_last_chat_response_len(
            infer.get_last_append_token_count());

        if (!infer.append("char", log_entry)) {
            fprintf(stderr, "Couldn't append chatlog\n");
            fflush(stderr);
            return false;
        }
        infer.commit("char");

        print_status(
            prompt_max_len, i, log_entry, prun_ctx, conversation, infer);

        //        float chat_fraction = ((float)i) /
        //        ((float)conversation.chat_turns()); int passed_time =
        //        time(NULL) - benchmark_start_time; float passed_tests_f =
        //            (((float)(prun_ctx.cur_test_nr - 1)) + chat_fraction);
        //        float time_per_test =
        //            passed_tests_f > 0.01 ? ((float)passed_time) /
        //            passed_tests_f : 0.0;
        //        float remaining =
        //            time_per_test * (((float)prun_ctx.total_tests) -
        //            passed_tests_f);
        //        remaining /= 60.0;
        //
        //        float passed_time_mins = ((float)passed_time) / 60.0;
        //
        //        std::string print_gen = process_text_for_console(log_entry);
        //        printf(
        //            "[test_id=%s, eta=%5.1fm, t=%5.1fm, seed=%ld, test_nr=%d,
        //            " "total_tests=%d, turn=(%d)%d/%d, plen(%d)=%d/%d,
        //            pusr=%d, " "pchr=%d]: %s\n", prun_ctx.test_id.c_str(),
        //            remaining,
        //            passed_time_mins,
        //            prun_ctx.seed,
        //            prun_ctx.cur_test_nr,
        //            prun_ctx.total_tests,
        //            conversation.shifted_chatlog_count(),
        //            i + 1,
        //            conversation.chat_turns(),
        //            conversation.get_total_token_count(),
        //            infer.current_max_seq_token_count(),
        //            prompt_max_len,
        //            infer.get_sequence_token_count("user"),
        //            infer.get_sequence_token_count("char"),
        //            print_gen.c_str());
        //        fflush(stdout);

        if (completion.size() == 0) {
            end_reason = "empty response";
            break;
        }

        is_user = !is_user;
    }

    llama_sampling_params &sparams = infer.sparams;

    std::string model_file = params.model.c_str();
    model_file = model_file.substr(model_file.find_last_of("/\\") + 1);
    conversation.log_chatlog_to_file(
        prun_ctx.seed, model_file, prompt_test, sparams);

    json prompt_collection;
    prompt_collection["sampler"] = sparams_to_json(sparams);
    prompt_collection["char"] = infer.get_sequence_text("user");
    prompt_collection["user"] = infer.get_sequence_text("char");
    prompt_collection["chatlog"] = conversation.chatlog_as_json();
    prompt_collection["rerolls"] = conversation.rerolls_as_json();
    prompt_collection["raw_chatlog"] = conversation.raw_chatlog_as_json();
    prompt_collection["text_chatlog"] = conversation.text_chatlog_as_json();
    prompt_collection["max_token_count"] = infer.current_max_seq_token_count();
    prompt_collection["rerolled_broken_quotes"] =
        conversation.rerolled_broken_quotes;
    prompt_collection["rerolled_empty_replies"] =
        conversation.rerolled_empty_replies;
    prompt_collection["end_reason"] = end_reason;

    j_resps.push_back(make_response(prun_ctx,
                                    conversation.chatlog_text(),
                                    infer.current_max_seq_token_count(),
                                    prompt_collection));

    return true;
}

std::vector<int64_t> get_seeds_for_test(int64_t default_seed,
                                        const json &prompt_runner_conf,
                                        const json &prompt_test);

std::vector<int64_t> get_seeds_for_test(int64_t default_seed,
                                        const json &prompt_runner_conf,
                                        const json &prompt_test) {
    std::vector<int64_t> seeds;
    if (prompt_test.find("seeds") != prompt_test.end()) {
        for (const auto &seed_value : prompt_test["seeds"]) {
            int64_t seed = seed_value;
            seeds.push_back(seed);
        }
    } else if (prompt_runner_conf.find("seeds") != prompt_runner_conf.end()) {
        for (const auto &seed_value : prompt_runner_conf["seeds"]) {
            int64_t seed = seed_value;
            seeds.push_back(seed);
        }
    } else {
        seeds.push_back(default_seed);
    }

    return seeds;
}

size_t get_total_test_count(const json &prompt_runner_conf);

size_t get_total_test_count(const json &prompt_runner_conf) {
    if (prompt_runner_conf.find("prompt_tests") == prompt_runner_conf.end()) {
        return 0;
    }

    size_t count = 0;

    for (const auto &prompt_test : prompt_runner_conf["prompt_tests"]) {
        auto seeds = get_seeds_for_test(1, prompt_runner_conf, prompt_test);
        count += seeds.size();
    }

    return count;
}

int main(int argc, char **argv) {
    gpt_params params;

    g_verbose = false;

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    llama_sampling_params sparams = params.sparams;

    if (!sparams.grammar.empty()) {
        grammar_parser::parse_state parsed_grammar;
        parsed_grammar = grammar_parser::parse(sparams.grammar.c_str());

        fprintf(stderr, "%s: grammar:\n", __func__);
        grammar_parser::print_grammar(stderr, parsed_grammar);
        fprintf(stderr, "\n");
        fflush(stderr);
    }

    SampleParamSource sampler_source(sparams);

    const json prompt_runner_conf = json::parse(params.prompt);

    if (prompt_runner_conf.find("prompt_tests") == prompt_runner_conf.end()) {
        fprintf(stderr, "**********\n");
        fprintf(stderr, "ERROR: No prompt_tests in prompt json!\n");
        fprintf(stderr, "**********\n");

        return 1;
    }

    if (prompt_runner_conf.find("samplers") != prompt_runner_conf.end()) {
        sampler_source.set_sampler_lib(prompt_runner_conf["samplers"]);
    }

    json selected_tests;
    if (params.input_prefix != "") {
        selected_tests = json::parse(params.input_prefix);
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

    llama_backend_init(params.numa);

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
            __func__,
            n_ctx_train,
            n_ctx);
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
        fprintf(stderr,
                "system_info: n_threads = %d / %d | %s\n",
                params.n_threads,
                std::thread::hardware_concurrency(),
                llama_print_system_info());
    }

    json j_resps;
    json j_tok_resps;

    bool first = true;

    bool record_next_token_info =
        prompt_runner_conf.value("record_next_token_info", false);

    benchmark_start_time = time(NULL);

    fprintf(stderr, "PROMPT-RUNNER-START\n");
    fflush(stderr);

    TextReplacer replacer(prompt_runner_conf);

    PromptRunContext prun_ctx;
    prun_ctx.total_tests = get_total_test_count(prompt_runner_conf);

    if (prun_ctx.total_tests == 0) {
        fprintf(stderr, "No \"prompt_tests\" defined in the prompt json!\n");
        return 1;
    }

    g_verbose = params.verbose_prompt;

    if (selected_tests.size() > 0) {
        for (const auto &sel_id : selected_tests) {
            std::string substr_id = sel_id;
            printf("selected test substring: %s\n", substr_id.c_str());
        }
    }

    for (const auto &prompt_test : prompt_runner_conf["prompt_tests"]) {
        json expected;
        if (prompt_test.find("expected") != prompt_test.end()) {
            expected = prompt_test["expected"];
        }
        prun_ctx.expected = expected;

        prun_ctx.test_id = prompt_test.value("id", "unknown_test_id");

        bool disabled = prompt_test.value("disabled", false);
        if (disabled) {
            printf("Disabled test: %s\n", prun_ctx.test_id.c_str());
            continue;
        }

        if (selected_tests.size() > 0) {
            bool any_matched = false;
            for (const auto &sel_id : selected_tests) {
                std::string substr_id = sel_id;
                if (prun_ctx.test_id.find(substr_id) != std::string::npos) {
                    any_matched = true;
                }
            }
            if (!any_matched) {
                printf("Skipped unselected test: %s\n",
                       prun_ctx.test_id.c_str());
                fflush(stdout);
                continue;
            }
        }

        fflush(stdout);

        TextReplacer test_replacer = replacer;
        test_replacer.merge_replacements(prompt_test);

        auto seeds =
            get_seeds_for_test(params.seed, prompt_runner_conf, prompt_test);

        for (const auto &seed_value : seeds) {
            prun_ctx.seed = seed_value;
            llama_set_rng_seed(ctx, prun_ctx.seed);

            prun_ctx.cur_test_nr += 1;

            if (params.prompt.empty()) {
                fprintf(stderr, "No prompt given!");
                return 1;
            }

            llama_sampling_params test_sparams = sparams;

            if (prompt_test.find("sampler") != prompt_test.end()) {
                json_to_sparams(prompt_test["sampler"], test_sparams);
            }

            if (first) {
                fprintf(stderr,
                        "sampling: \n%s\n",
                        llama_sampling_print(test_sparams).c_str());
            }

            if (prompt_test.find("chat") != prompt_test.end()) {
                Inference infer(test_sparams, params.n_batch);

                if (!chatlog_generator(prun_ctx,
                                       params,
                                       infer,
                                       replacer,
                                       prompt_runner_conf,
                                       prompt_test,
                                       j_resps)) {
                    return 1;
                }

            } else if (prompt_test.find("script") != prompt_test.end()) {
                Inference infer(test_sparams, params.n_batch);

                CompletionScript cscript;
                cscript.from_json(prompt_test);

                json node_results;

                int node_count = 0;
                for (auto &node : cscript.nodes) {
                    if (node.skip) continue;
                    node_count += 1;
                };

                int i = 0;
                for (auto &node : cscript.nodes) {
                    if (node.skip) {
                        continue;
                    }

                    infer.reset_seed(prun_ctx.seed + node.index);

                    auto com = infer.complete_node(
                        "main", node, sampler_source, test_replacer);
                    if (com.error) {
                        return 1;
                    }

                    std::string res = node.result_to_json().dump(
                        2, ' ', false, json::error_handler_t::replace);
                    // d// printf("RES: %s\n", res.c_str());
                    node_results.push_back(node.result_to_json());

                    print_script_status(
                        i, node_count, node.result_string, prun_ctx, infer);

                    i++;
                }

                //                json prompt_collection;
                // prompt_collection["sampler"] = sparams_to_json(sparams);
                // prompt_collection["char"] = infer.get_sequence_text("user");
                // prompt_collection["user"] = infer.get_sequence_text("char");
                // prompt_collection["chatlog"] =
                // conversation.chatlog_as_json(); prompt_collection["rerolls"]
                // = conversation.rerolls_as_json();
                // prompt_collection["raw_chatlog"] =
                // conversation.raw_chatlog_as_json();
                // prompt_collection["text_chatlog"] =
                // conversation.text_chatlog_as_json();
                //                prompt_collection["nodes"] = node_results;

                j_resps.push_back(
                    make_response(prun_ctx,
                                  infer.get_sequence_text("main"),
                                  infer.current_max_seq_token_count(),
                                  node_results));

                // TODO: Append to j_resps
            }

            first = false;
        }
    }

    llama_print_timings(ctx);

    std::string model_file = params.model.c_str();
    model_file = model_file.substr(model_file.find_last_of("/\\") + 1);
    printf("model: %s\n", model_file.c_str());

    json j_params = sparams_to_json(sparams);
    j_params["rope_freq_base"] = params.rope_freq_base;
    j_params["rope_freq_scale"] = params.rope_freq_scale;

    json build_info;
    build_info["build"] = LLAMA_BUILD_NUMBER;
    build_info["commit"] = LLAMA_COMMIT;
    build_info["compiler"] = LLAMA_COMPILER;
    build_info["build_target"] = LLAMA_BUILD_TARGET;
    build_info["runner_version"] = "v0.5.0";

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
