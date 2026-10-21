const std = @import("std");

const types = @import("../../types.zig");

const GGMLDataType = types.GGMLDataType;
const DeviceType = types.DeviceType;

pub fn dataTypeEnumRawValue(data_type: GGMLDataType) c_int {
    return @intFromEnum(data_type);
}

pub fn deviceTypeEnumRawValue(dev: DeviceType) c_int {
    return @intFromEnum(dev);
}
