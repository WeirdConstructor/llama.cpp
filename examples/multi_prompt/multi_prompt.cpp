#include "common.h"
#include "llama.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

static llama_context **g_ctx;
static llama_model **g_model;

llama_batch make_token_batch(const std::string &text, int seq_id, int p0 = 0);
llama_batch make_token_batch(const std::string &text, int seq_id, int p0) {
    std::vector<llama_token> toks;
    toks = ::llama_tokenize(*g_model, text.c_str(), true);

    auto batch = llama_batch_init(toks.size(), 0, 1);

    for (int i = 0; i < (int)toks.size(); i++) {
        llama_batch_add(batch, toks[i], p0 + i, {seq_id}, false);
    }

    return batch;
}

int main(int argc, char **argv) {
    gpt_params params;

    if (argc == 1 || argv[1][0] == '-') {
        printf("usage: %s MODEL_PATH [PROMPT] [PARALLEL] [LEN] [NGL]\n",
               argv[0]);
        return 1;
    }

    if (argc >= 2) {
        params.model = argv[1];
    }

    int decode_order_batch2_first = 0;
    if (argc >= 3) {
        std::string s = argv[2];
        if (s == "1")
            decode_order_batch2_first = 1;
        else
            decode_order_batch2_first = 0;
    }

    // init LLM

    llama_backend_init();

    // initialize the model

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 4;

    llama_model *model =
        llama_load_model_from_file(params.model.c_str(), model_params);

    if (model == NULL) {
        fprintf(stderr, "%s: error: unable to load model\n", __func__);
        return 1;
    }

    g_model = &model;

    llama_context_params ctx_params = llama_context_default_params();

    ctx_params.seed = 1234;
    ctx_params.n_ctx = 512;
    ctx_params.n_batch = 256;
    ctx_params.n_threads = params.n_threads;
    ctx_params.n_threads_batch = params.n_threads_batch == -1
                                     ? params.n_threads
                                     : params.n_threads_batch;

    llama_context *ctx = llama_new_context_with_model(model, ctx_params);

    if (ctx == NULL) {
        fprintf(stderr,
                "%s: error: failed to create the llama_context\n",
                __func__);
        return 1;
    }

    g_ctx = &ctx;

    // tokenize the first prompt for sequence id=0
    int seq_prompt1 = 0;
//    auto batch1 = make_token_batch("Add some random text just to have more than 32 tokens in the batch. Add some random text just to have more than 32 tokens in the batch. Add some random text just to have more than 32 tokens in the batch. The quick fast brown and yellow fox jumps",
    auto batch1 = make_token_batch("01234567 01234567 012345670120123456789050 The quick fast brown and yellow fox jumps",
                                   seq_prompt1);
    int batch1_len = batch1.n_tokens;

    // tokenize the second prompt for sequence id=1
    int seq_prompt2 = 1;
    auto batch2 = make_token_batch(
        "XIFDEIFEIFE0123456780123456780123456",
        seq_prompt2,
        300);
    // int batch2_len = batch2.n_tokens;

    if (decode_order_batch2_first) {
        // load sequence id=1 with [p0=300, p1=386)
        if (llama_decode(ctx, batch2)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return false;
        }
        // load sequence id=0 with [p0=0, p1=11)
        if (llama_decode(ctx, batch1)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return false;
        }
    } else {
        // load sequence id=0 with [p0=0, p1=11)
        if (llama_decode(ctx, batch1)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return false;
        }
        // load sequence id=1 with [p0=300, p1=386)
        if (llama_decode(ctx, batch2)) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return false;
        }
    }

    // put new batch to the end of the mid, into sequence of prompt1:
    int seq0_p1 = batch1_len;
    auto batch_compl = make_token_batch(" over the", seq_prompt1, seq0_p1);
    batch_compl.logits[batch_compl.n_tokens - 1] = true;

    if (llama_decode(ctx, batch_compl)) {
        fprintf(stderr, "%s : failed to eval\n", __func__);
        return false;
    }

//    llama_kv_cache_debug_print(ctx, "Z");

    // start sampling from the last position of [prompt1 + mid + tokens(" the")]
    seq0_p1 = batch1_len + batch_compl.n_tokens;

    llama_sampling_params sparams;
    llama_sampling_context *ctx_sampling = llama_sampling_init(sparams);

    // sample index is the position in the recent batch. First iteration this is the
    // last token of " the". In the other iterations it's always 0.
    int sample_idx = batch_compl.n_tokens - 1;

    int n_remain = 30;
    int p0_compl = seq0_p1;
    std::string completion;
    while (n_remain-- > 0) {
        llama_token tok =
            llama_sampling_sample(ctx_sampling, ctx, nullptr, sample_idx);

        const std::string piece = llama_token_to_piece(*g_ctx, tok);
        completion += piece;

        if (llama_decode(ctx, llama_batch_get_one(&tok, 1, seq0_p1, seq_prompt1))) {
            fprintf(stderr, "%s : failed to eval\n", __func__);
            return false;
        }

        seq0_p1 += 1;
        sample_idx = 0;
    }
    int p1_compl = seq0_p1;
    printf("Completion [p0=%d,p1=%d): {%s}\n", p0_compl, p1_compl, completion.c_str());

//    llama_kv_cache_debug_print(ctx, "E");

    llama_batch_free(batch1);
    llama_batch_free(batch2);
    llama_batch_free(batch_compl);
    llama_free(ctx);
    llama_free_model(model);

    llama_backend_free();

    return 0;
}
