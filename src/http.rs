use std::{
    borrow::Cow, io::{self, Read, Result, Write}
};
use crate::aruntime;
use crate::ffi;
fn get_req(path: &str) -> String {
    format!(
        "GET {path} HTTP/1.1\r\n\
             Host: localhost\r\n\
             Connection: close\r\n\
             \r\n"
    )
}

pub fn http_get(addr: &str,path:&str)->Result<String>{
    let request = get_req(&path);
    let mut stream = std::net::TcpStream::connect(addr)?;
    stream.set_nonblocking(true)?;
    stream.write_all(request.as_bytes())?;
    aruntime::async_wait(&stream, ffi::EPOLLIN | ffi::EPOLLET)?;
    let mut data = vec![0u8; 4096];
    let mut res_body = String::new();
    loop {
        match stream.read(&mut data) {
            Ok(n) if n == 0 => {
                break;
            },
            Ok(n) => {
                let txt = String::from_utf8_lossy(&data[..n]);
                res_body.push_str(&txt);
            }
            Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
            // this was not in the book example, but it's a error condition
            // you probably want to handle in some way (either by breaking
            // out of the loop or trying a new read call immediately)
            Err(e) if e.kind() == io::ErrorKind::Interrupted => break,
            Err(e) => {
                return Err(e);
            }
        }
    }
    return Ok(res_body);
}