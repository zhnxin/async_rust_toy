#![feature(naked_functions)]
use std::arch::{asm, naked_asm};
use std::{
    io::Result,
    net::TcpStream,
};
use crate::ffi::Event;
use crate::poll;
// 栈大小2M
const DEFAULT_STACK_SIZE: usize = 1024 * 1024 * 2;
// 默认4个协程
const MAX_THREADS: usize = 4;

#[derive(PartialEq, Eq, Debug)]
enum State {
    Available,// 空闲，可用线程
    Running, //正在执行
    Ready, // 允许重新执行
    Pending, // 等待IO就绪
}

struct Thread {
    stack: Vec<u8>,// 栈数据
    ctx: ThreadContext, // 上下文
    state: State, // 协程状态
}

#[derive(Debug, Default)]
#[repr(C)]
struct ThreadContext {
    rsp: u64,
    r15: u64,
    r14: u64,
    r13: u64,
    r12: u64,
    rbx: u64,
    rbp: u64,
}

impl Thread {
    fn new() -> Self {
        Thread {
            stack: vec![0_u8; DEFAULT_STACK_SIZE],
            ctx: ThreadContext::default(),
            state: State::Available,
        }
    }
    fn new_with_state(state:State) -> Self {
        Thread {
            stack: vec![0_u8; DEFAULT_STACK_SIZE],
            ctx: ThreadContext::default(),
            state: state,
        }
    }
}

static mut RUNTIME: usize = 0;
pub struct Runtime {
    threads: Vec<Thread>,
    current: usize,
    event_poll:poll::Poll,
}
impl Runtime {
    pub fn new() -> Self {
        let base_thread = Thread::new_with_state(State::Running);

        let mut threads = vec![base_thread];
        let mut available_threads: Vec<Thread> = (1..MAX_THREADS).map(|_| Thread::new()).collect();
        threads.append(&mut available_threads);

        Runtime {
            threads,
            current: 0,
            event_poll: poll::Poll::new().unwrap(),// 不推荐的写法，初始化失败直接panic
        }
    }
    pub fn async_wait(&mut self,source: &TcpStream, interests: i32)-> Result<()> {
        self.event_poll.registry().register(source, self.current, interests)?;
        self.threads[self.current].state = State::Pending;
        self.t_yield();
        Ok(())
    }

    pub fn init(&self) {
        unsafe {
            let r_ptr: *const Runtime = self;
            RUNTIME = r_ptr as usize;
        }
    }

    pub fn run(&mut self) -> ! {
        while self.t_yield() {}
        std::process::exit(0);
    }

    fn t_return(&mut self) {
        // 如果当前协程不是base协程，将状态改为Available
        if self.current != 0 {
            self.threads[self.current].state = State::Available;
            self.t_yield();
        }
    }
    fn poll_event(&mut self) -> bool{
        if self.threads.iter().all(|s|s.state != State::Pending){
            return false;
        }
        let mut events:Vec<Event> = Vec::with_capacity(MAX_THREADS);
        self.event_poll.poll(&mut events, None).unwrap();// ugly code：遇到异常直接panic
        for event in events {
            let index = event.token();
            self.threads[index].state = State::Ready;
        }
        return true;
    }
    // 所有线程均为空
    fn is_all_complated(&self) -> bool{
        self.threads.iter().all(|t| t.state == State::Available)
    }
    fn robin_get_ready_thread(&self) -> (usize,bool){
        let mut pos = self.current;
        while self.threads[pos].state != State::Ready {
            pos += 1;
            if pos == self.threads.len() {
                pos = 0;
            }
            if pos == self.current {
                return (self.current,false);
            }
        }
        return (pos,true);
    }

    #[inline(never)]
    fn t_yield(&mut self) -> bool {
        // Robin算法获取下一个可执行的线程
        let (mut pos,mut ok) = self.robin_get_ready_thread();
        if !ok {
            // 没有就绪的线程
            if !self.poll_event() {
                // 没有事件需等待
                return false;
            }
            (pos,ok) = self.robin_get_ready_thread();
        }
        if !ok {
            return false;
        }
        // 当前线程为State::Running，将状态改为可执行
        // State::Available 说明线程已执行完成
        // State::Pending 说明线程在等待数据
        if self.threads[self.current].state == State::Running {
            self.threads[self.current].state = State::Ready;
        }
        // 切换线程状态
        self.threads[pos].state = State::Running;
        let old_pos = self.current;
        self.current = pos;

        unsafe {
            let old: *mut ThreadContext = &mut self.threads[old_pos].ctx;
            let new: *const ThreadContext = &self.threads[pos].ctx;
            asm!("call switch", in("rdi") old, in("rsi") new, clobber_abi("C"));
        }
        self.threads.len() > 0
    }

    pub fn spawn(&mut self, f: fn()) {
        let available = if let Some(thread) = self
            .threads
            .iter_mut()
            .find(|t| t.state == State::Available)
        {
            thread
        } else {
            self.threads.push(Thread::new());
            self.threads.last_mut().expect("failed to create new thread")
        };

        let size = available.stack.len();

        unsafe {
            // 将需要执行的函数入栈
            let s_ptr = available.stack.as_mut_ptr().offset(size as isize);
            let s_ptr = (s_ptr as usize & !15) as *mut u8;
            std::ptr::write(s_ptr.offset(-16) as *mut u64, guard as u64);
            std::ptr::write(s_ptr.offset(-24) as *mut u64, skip as u64);
            std::ptr::write(s_ptr.offset(-32) as *mut u64, f as u64);
            available.ctx.rsp = s_ptr.offset(-32) as u64;
        }
        available.state = State::Ready;
    }
} 

pub fn async_wait(source: &TcpStream, interests: i32)-> Result<()>{
    unsafe {
        let rt_ptr = RUNTIME as *mut Runtime;
        (*rt_ptr).async_wait(source,interests)?;
    };
    Ok(())
}

fn guard() {
    unsafe {
        let rt_ptr = RUNTIME as *mut Runtime;
        (*rt_ptr).t_return();
    };
}

#[unsafe(naked)]
unsafe extern "C" fn skip() {
    naked_asm!("ret")
}

pub fn yield_thread() {
    unsafe {
        let rt_ptr = RUNTIME as *mut Runtime;
        (*rt_ptr).t_yield();
    };
}

#[unsafe(naked)]
#[no_mangle]
#[cfg_attr(target_os = "macos", export_name = "\x01switch")] // see: How-to-MacOS-M.md for explanation
unsafe extern "C" fn switch() {
    naked_asm!(
        "mov [rdi + 0x00], rsp",
        "mov [rdi + 0x08], r15",
        "mov [rdi + 0x10], r14",
        "mov [rdi + 0x18], r13",
        "mov [rdi + 0x20], r12",
        "mov [rdi + 0x28], rbx",
        "mov [rdi + 0x30], rbp",
        "mov rsp, [rsi + 0x00]",
        "mov r15, [rsi + 0x08]",
        "mov r14, [rsi + 0x10]",
        "mov r13, [rsi + 0x18]",
        "mov r12, [rsi + 0x20]",
        "mov rbx, [rsi + 0x28]",
        "mov rbp, [rsi + 0x30]",
        "ret"
    );
}
