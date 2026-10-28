//! DeepSeek-V4.1-Flash on CUDA in Zig, tensor parallel over two GB10s: the served Python family's kernels, host first.

pub const Config = @import("config.zig").Config;
pub const native = @import("native.zig");
pub const rank_cache = @import("rank_cache.zig");
pub const plan = @import("plan.zig");
pub const engram = @import("engram.zig");
pub const picture = @import("picture.zig");
pub const cublaslt = @import("cublaslt.zig");
pub const fmha = @import("fmha.zig");
pub const vit = @import("vit.zig");
pub const vision = @import("vision.zig");
pub const pil = @import("pil.zig");
pub const checkpoint = @import("checkpoint.zig");
pub const weights = @import("weights.zig");
pub const engram_io = @import("engram_io.zig");
pub const engram_aio = @import("engram_aio.zig");
pub const link = @import("link.zig");
pub const comm = @import("comm.zig");
pub const tri = @import("tri.zig");
pub const tri_basic = @import("tri_basic.zig");
pub const cublas = @import("cublas.zig");
pub const elf = @import("elf.zig");
pub const exl3_linear = @import("exl3_linear.zig");
pub const tri_norm = @import("tri_norm.zig");
pub const tri_index = @import("tri_index.zig");
pub const tri_attn = @import("tri_attn.zig");
pub const tri_markov = @import("tri_markov.zig");
pub const tri_hc = @import("tri_hc.zig");
pub const exl3_prefill = @import("exl3_prefill.zig");
pub const ops = @import("ops.zig");
pub const exl3_experts = @import("exl3_experts.zig");
pub const exl3_experts_rec = @import("exl3_experts_rec.zig");
pub const prompt = @import("prompt.zig");
pub const exact = @import("exact.zig");
pub const rdma_seq = @import("rdma_seq.zig");
pub const rdma = @import("rdma.zig");
pub const exl3_experts2d = @import("exl3_experts2d.zig");
pub const exl3_experts2d_rec = @import("exl3_experts2d_rec.zig");
pub const prompt2d = @import("prompt2d.zig");
pub const ring2d = @import("ring2d.zig");
pub const text = @import("text.zig");
pub const round = @import("round.zig");
pub const round2d = @import("round2d.zig");
pub const draft2d = @import("draft2d.zig");
pub const markov_tokens = @import("markov_tokens.zig");
pub const draft = @import("draft.zig");
pub const sampling = @import("sampling.zig");
pub const model = @import("model.zig");
pub const lanes = @import("lanes.zig");
// PETABRIDGE (k2a-pb1): spill v2 host modules (MiaAI 0016/0019 tip), compiled and tested, not yet wired (P3)
pub const spill_state = @import("spill_state.zig");
pub const rank_wire = @import("rank_wire.zig");
pub const fingerprint = @import("fingerprint.zig");
pub const rank_identity = @import("rank_identity.zig");
pub const parts = @import("parts.zig");
// PETABRIDGE (k2a-pb1) spill P6: the tier itself (wired into lanes, native and lanes_main; off unless asked)
pub const spill_tier = @import("spill_tier.zig");

test {
    // every declaration of every file, so functions no test calls are still compiled
    const refAll = @import("std").testing.refAllDecls;
    refAll(@import("config.zig"));
    refAll(native);
    refAll(rank_cache);
    refAll(plan);
    refAll(engram);
    refAll(checkpoint);
    refAll(weights);
    refAll(engram_io);
    refAll(engram_aio);
    refAll(link);
    refAll(comm);
    refAll(tri);
    refAll(tri_basic);
    refAll(cublas);
    refAll(elf);
    refAll(exl3_linear);
    refAll(tri_norm);
    refAll(tri_index);
    refAll(tri_attn);
    refAll(tri_markov);
    refAll(tri_hc);
    refAll(exl3_prefill);
    refAll(ops);
    refAll(exl3_experts);
    refAll(exl3_experts_rec);
    refAll(prompt);
    refAll(exact);
    refAll(rdma_seq);
    refAll(rdma);
    refAll(exl3_experts2d);
    refAll(exl3_experts2d_rec);
    refAll(prompt2d);
    refAll(ring2d);
    refAll(text);
    refAll(round);
    refAll(round2d);
    refAll(draft2d);
    refAll(markov_tokens);
    refAll(draft);
    refAll(sampling);
    refAll(model);
    refAll(lanes);
    // PETABRIDGE (k2a-pb1): the spill v2 host modules' tests (P3); MiaAI's model/lanes/native refs wait for P4-P6
    refAll(spill_state);
    refAll(rank_wire);
    refAll(fingerprint);
    refAll(rank_identity);
    refAll(parts);
    // PETABRIDGE (k2a-pb1): MiaAI 0016's compile check, verbatim: rank_identity.gather over our real Options and Model
    _ = &struct {
        fn f(gpa: @import("std").mem.Allocator, io: @import("std").Io, o: model.Options, m: *model.Model, x: rank_identity.Extra) !rank_identity.Gathered {
            return rank_identity.gather(gpa, io, o, m, x);
        }
    }.f;
    // PETABRIDGE (k2a-pb1): the spill state's model side (P4; MiaAI 0016's refs, its tier/lanes/native ones wait for
    // P5-P6): its methods and the packer over the device planes, compiled although only open's check calls them yet
    refAll(model.Model.SpillDev);
    refAll(spill_state.Packer(model.Model.SpillDev));
    _ = &model.Model.spillLayout;
    _ = &model.Model.spillGenerations;
    _ = &model.Model.spillPacker;
    _ = &model.Model.spillSnapTail;
    _ = &model.Model.spillClearSnapTail;
    _ = &model.Model.spillCheckResumed;
    _ = &model.Model.spillSelfCheck;
    _ = &model.Model.spillCheckLine;
    // PETABRIDGE (k2a-pb1) spill P6: the tier (its tests: spill_tier_test.zig, loopback, worlds 2 and 3) and 0016's
    // lanes/native refs: the Leader and Follower over the real device, the lanes' family and follower side
    refAll(spill_tier);
    refAll(lanes.SpillTier);
    refAll(lanes.SpillFollower);
    _ = &lanes.Lanes.spillFamily;
    _ = &lanes.Lanes.spillWire;
    _ = &lanes.followWith;
    _ = &native.openSpill;
    _ = &native.followerSpill;
    _ = &native.rankExchange;
    _ = @import("spill_p6_test.zig");
}
