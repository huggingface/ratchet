use derive_new::new;
#[macro_export]
macro_rules! wgs {
    ($x:expr, $y:expr, $z:expr) => {
        $crate::gpu::WorkgroupSize::new($x, $y, $z)
    };
}

#[derive(Debug, Clone, new, PartialEq, Eq, Hash)]
pub struct WorkgroupSize {
    x: u32,
    y: u32,
    z: u32,
}

impl WorkgroupSize {
    pub fn x(&self) -> u32 {
        self.x
    }

    pub fn y(&self) -> u32 {
        self.y
    }

    pub fn z(&self) -> u32 {
        self.z
    }

    pub fn as_slice(&self) -> [u32; 3] {
        [self.x, self.y, self.z]
    }

    pub fn total(&self) -> u32 {
        self.x * self.y * self.z
    }
}

impl From<WorkgroupSize> for glam::UVec3 {
    fn from(val: WorkgroupSize) -> Self {
        glam::UVec3::new(val.x, val.y, val.z)
    }
}

impl ToString for WorkgroupSize {
    fn to_string(&self) -> String {
        format!("{},{},{}", self.x, self.y, self.z)
    }
}

impl Default for WorkgroupSize {
    fn default() -> Self {
        Self::new(1, 1, 1)
    }
}

#[macro_export]
macro_rules! wgc {
    ($x:expr, $y:expr, $z:expr) => {
        $crate::gpu::WorkgroupCount::new($x, $y, $z)
    };
}

#[derive(Debug, Clone, new, PartialEq, Eq, Hash)]
pub struct WorkgroupCount {
    x: u32,
    y: u32,
    z: u32,
}

impl WorkgroupCount {
    pub fn x(&self) -> u32 {
        self.x
    }

    pub fn y(&self) -> u32 {
        self.y
    }

    pub fn z(&self) -> u32 {
        self.z
    }

    pub fn as_slice(&self) -> [u32; 3] {
        [self.x, self.y, self.z]
    }

    pub fn total_count(&self) -> u32 {
        self.x * self.y * self.z
    }
}

impl Default for WorkgroupCount {
    fn default() -> Self {
        Self::new(1, 1, 1)
    }
}

#[derive(Debug, new, Clone, Default)]
pub struct GpuWorkload {
    workgroup_count: WorkgroupCount,
    workgroup_size: WorkgroupSize,
}

impl GpuWorkload {
    pub fn workgroup_count(&self) -> &WorkgroupCount {
        &self.workgroup_count
    }

    pub fn workgroup_size(&self) -> &WorkgroupSize {
        &self.workgroup_size
    }

    pub fn counts(&self) -> [u32; 3] {
        self.workgroup_count.as_slice()
    }

    pub fn sizes(&self) -> [u32; 3] {
        self.workgroup_size.as_slice()
    }

    pub fn prev_power_of_two(n: usize) -> usize {
        1 << ((8 * std::mem::size_of::<usize>()) - n.leading_zeros() as usize - 1)
    }

    pub fn next_power_of_two(n: usize) -> usize {
        1 << (8 * std::mem::size_of::<usize>() - n.leading_zeros() as usize)
    }
}

impl GpuWorkload {
    pub const MAX_WORKGROUP_SIZE_X: usize = 256;
    pub const MAX_WORKGROUP_SIZE_Y: usize = 256;
    pub const MAX_WORKGROUP_SIZE_Z: usize = 64;
    pub const MAX_COMPUTE_WORKGROUPS_PER_DIMENSION: usize = 65535;
    pub const MAX_COMPUTE_INVOCATIONS_PER_WORKGROUP: usize = 256;

    pub fn div_ceil(num: usize, div: usize) -> usize {
        num / div + (num % div != 0) as usize
    }
}
