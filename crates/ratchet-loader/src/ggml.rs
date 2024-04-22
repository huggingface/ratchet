use byteorder::ReadBytesExt;
use std::mem::MaybeUninit;

trait ReadBytesCustom: ReadBytesExt {
    /// Extends to read an exact number of bytes.
    fn read_bytes_with_len(&mut self, len: usize) -> std::io::Result<Vec<u8>> {
        let mut buf: Vec<MaybeUninit<u8>> = Vec::with_capacity(len);
        unsafe {
            buf.set_len(len);
        }
        let buf_slice = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut u8, len) };
        self.read_exact(buf_slice)?;
        let buf = unsafe { std::mem::transmute::<_, Vec<u8>>(buf) };
        Ok(buf)
    }
}
impl<T: std::io::BufRead> ReadBytesCustom for T {}
