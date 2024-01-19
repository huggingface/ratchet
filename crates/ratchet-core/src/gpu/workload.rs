use derive_new::new;

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

    /// Divide a number by the indicated dividend, then round up to the next multiple of the dividend if there is a rest.
    pub fn div_ceil(num: usize, div: usize) -> usize {
        num / div + (num % div != 0) as usize
    }
}

impl Default for WorkgroupCount {
    fn default() -> Self {
        Self::new(1, 1, 1)
    }
}
