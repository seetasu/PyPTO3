#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/costmodel/perf_sim/launch.hpp>
#include <cstdint>
#include "../ptoas/attn_swpipe.cpp"

void run_attn_swpipe()
{
    static int64_t workspace[1024]{};
    static bfloat16_t key_cache[1048576]{};
    static bfloat16_t query[32768]{};
    static bfloat16_t value_cache[1048576]{};
    static int32_t seq_lens[2]{512, 513};
    static float inv_rms[2]{1, 1};
    static int32_t slot_mapping[2]{};
    static float rope_cos[524288]{};
    static float rope_sin[524288]{};
    static float k_proj[2048]{};
    static float k_norm_w[128]{};
    static float v_proj[2048]{};
    static float q_proj[10240]{};
    static float q_norm_w[128]{};
    static int32_t block_table[65536]{};
    static float score_transfer[589824]{};
    static bfloat16_t probability_transfer[589824]{};
    static float pv_transfer[147456]{};
    static bfloat16_t out[32768]{};
    attn_swpipe_aic(workspace, key_cache, query, value_cache, seq_lens, inv_rms,
        slot_mapping, rope_cos, rope_sin, k_proj, k_norm_w, v_proj, q_proj,
        q_norm_w, block_table, score_transfer, probability_transfer,
        pv_transfer, out, 2, 0, 16, 0, 10368, 80, 2, 32, 0, 24);
    attn_swpipe_aiv(workspace, key_cache, query, value_cache, seq_lens, inv_rms,
        slot_mapping, rope_cos, rope_sin, k_proj, k_norm_w, v_proj, q_proj,
        q_norm_w, block_table, score_transfer, probability_transfer,
        pv_transfer, out, 2, 0, 16, 0, 10368, 80, 2, 32, 0, 24, 0);
}

int main()
{
    LAUNCH_KERNEL(run_attn_swpipe, , (1, nullptr, nullptr));
    return 0;
}
