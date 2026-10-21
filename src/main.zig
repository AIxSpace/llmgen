const llmgen = @import("llmgen.zig");

pub fn main() !void {
    try llmgen.run();
}

test {
    _ = @import("llmgen.zig");
}
