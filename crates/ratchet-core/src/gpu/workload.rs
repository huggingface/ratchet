use derive_new::new;
use inline_wgsl::wgsl;

use crate::KernelElement;

#[derive(Debug, Clone, new, PartialEq, Eq, Hash)]
pub struct WorkgroupSize {
    pub x: u32,
    pub y: u32,
    pub z: u32,
}

impl WorkgroupSize {
    pub fn product(&self) -> u32 {
        self.x * self.y * self.z
    }

    pub fn as_key(&self) -> String {
        format!("{}_{}_{}", self.x, self.y, self.z)
    }
}

#[macro_export]
macro_rules! wgs {
    ($x:expr, $y:expr, $z:expr) => {
        $crate::gpu::WorkgroupSize::new($x, $y, $z)
    };
}

impl std::fmt::Display for WorkgroupSize {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let WorkgroupSize { x, y, z } = self;
        write!(f, "{}", wgsl! { @compute @workgroup_size('x, 'y, 'z) })
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

    pub fn product(&self) -> u32 {
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

#[derive(Debug)]
pub struct Workload {
    pub workgroup_size: WorkgroupSize,
    pub workgroup_count: WorkgroupCount,
}

impl Workload {
    pub fn std(numel: usize, ke: KernelElement) -> Workload {
        let workgroup_size = wgs![8, 8, 1];

        let numel = numel / ke.as_size();
        let x_groups = WorkgroupCount::div_ceil(numel as _, workgroup_size.product() as _);
        let (x_groups, y_groups) = if x_groups > WorkgroupCount::MAX_WGS_PER_DIM {
            let y_groups = WorkgroupCount::div_ceil(x_groups, WorkgroupCount::MAX_WGS_PER_DIM);
            (WorkgroupCount::MAX_WGS_PER_DIM, y_groups)
        } else {
            (x_groups, 1)
        };

        Workload {
            workgroup_count: wgc![x_groups as _, y_groups as _, 1],
            workgroup_size,
        }
    }
}
