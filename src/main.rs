mod aruntime;
mod ffi;
mod http;
mod poll;

fn main() {
    let mut runtime = aruntime::Runtime::new();
    runtime.init();

    runtime.spawn(|| {
        println!("THREAD 1 STARTING");
        let id = 1;
        let host = "127.0.0.1:9091";
        for i in 0..15 {
            println!("thread: {} counter: {}", id, i);
            let url_path = format!("/sleep?mode=random&id={}_{}&sleep_time=500",id,i);
            match http::http_get(&host,&url_path){
                Ok(resp)=>{
                    println!("REVICE: {}",resp);
                }
                Err(e)=>{
                    println!("ERR: {}",e);
                }
            }
        }
        println!("THREAD 1 FINISHED");
    });

    runtime.spawn(|| {
        println!("THREAD 2 STARTING");
        let id = 2;
        let host = "127.0.0.1:9091";
        for i in 0..15 {
            println!("thread: {} counter: {}", id, i);
            let url_path = format!("/sleep?mode=random&id={}_{}&sleep_time=500",id,i);
            match http::http_get(&host,&url_path){
                Ok(resp)=>{
                    println!("REVICE: {}",resp);
                }
                Err(e)=>{
                    println!("ERR: {}",e);
                }
            }
        }
        println!("THREAD 2 FINISHED");
    });
    runtime.run();
}