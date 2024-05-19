use derive_new::new;

#[derive(Debug, Clone, new, PartialEq, Eq, Hash)]
pub struct WorkgroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

#[macro_export]
macro_rules! wgs {
    ($x:expr, $y:expr, $z:expr) => {
        $crate::gpu::WorkgroupSize::new($x, $y, $z)
    };
}

impl std::fmt::Display for WorkgroupSize {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(
            f,
            "@compute @workgroup_size({}, {}, {})",
            self.x, self.y, self.z
        )
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
    pub const MAX_WORKGROUP_SIZE_X: usize = 256;
    pub const MAX_WORKGROUP_SIZE_Y: usize = 256;
    pub const MAX_WORKGROUP_SIZE_Z: usize = 64;
    pub const MAX_WGS_PER_DIM: usize = 65535;
    pub const MAX_THREADS_PER_WG: usize = 256;

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

    /// Divide a number by the indicated dividend, then round up to the next multiple of the dividend if there is a rest.
    pub fn div_ceil(num: usize, div: usize) -> usize {
        num / div + (num % div != 0) as usize
    }
}

impl std::fmt::Display for WorkgroupCount {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "x{}_y{}_z{}", self.x, self.y, self.z)
    }
}

impl Default for WorkgroupCount {
    fn default() -> Self {
        Self::new(1, 1, 1)
    }
}
