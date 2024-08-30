// Adapted from https://github.com/huggingface/candle/blob/fc67d878bb4a25cbeba361d0a31290f14beb9344/candle-core/src/quantized/utils.rs

use half::f16;

use crate::error::Result;

pub trait ReadHalf {
    fn read_f16(&mut self) -> Result<f16>;
}

impl<R: std::io::Seek + std::io::Read> ReadHalf for R {
    fn read_f16(&mut self) -> Result<f16> {
        let mut d = [0u8; 2];
        self.read_exact(&mut d)?;
        let f16_value = half::f16::from_le_bytes(d);
        Ok(f16_value)
    }
}

pub trait WriteHalf {
    fn write_f16(&mut self, input: f16) -> Result<usize>;
}

impl<W: std::io::Seek + std::io::Write> WriteHalf for W {
    fn write_f16(&mut self, input: f16) -> Result<usize> {
        let bytes = input.to_le_bytes();
        let num_written = self.write(&bytes)?;
        Ok(num_written)
    }
}

pub trait ReadInto<Other> {
    fn read_u8s_into(&mut self, other: &mut Other, length: usize) -> Result<()>;
}

impl<R: std::io::Seek + std::io::Read, Other: std::io::Write> ReadInto<Other> for R {
    fn read_u8s_into(&mut self, other: &mut Other, length: usize) -> Result<()> {
        let mut temp = vec![0u8; length];
        self.read_exact(&mut temp)?;
        other.write_all(&temp)?;
        Ok(())
    }
}

pub trait ReadLen {
    fn read_len_bytes(&mut self, length: usize) -> Result<Vec<u8>>;
}

impl<R: std::io::Seek + std::io::Read> ReadLen for R {
    fn read_len_bytes(&mut self, length: usize) -> Result<Vec<u8>> {
        let mut temp = vec![0u8; length];
        self.read_exact(&mut temp)?;
        Ok(temp)
    }
}
