use std::{
    collections::{HashMap, VecDeque},
    ops::{Deref, DerefMut},
};

/// TODO
#[derive(Debug, Default)]
pub struct MyVec<T>(Vec<T>);

impl<T> Deref for MyVec<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for MyVec<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<T> MyVec<T> {
    /// TODO
    pub fn new() -> Self {
        Self(Vec::with_capacity(50))
    }

    /// TODO
    pub fn with_capacity(capacity: usize) -> Self {
        Self(Vec::with_capacity(capacity))
    }

    /// TODO
    pub fn push(&mut self, value: T) {
        let len = self.0.len();
        if len == self.0.capacity() {
            self.0.reserve_exact(len * 2);
        }

        self.0.push(value);
    }
}

/// TODO
pub struct VecStore<T> {
    bufs: VecDeque<MyVec<T>>,
}

impl<T> VecStore<T> {
    /// TODO
    pub fn new() -> Self {
        let mut bufs = VecDeque::with_capacity(700);
        (0..bufs.capacity()).for_each(|_| {
            bufs.push_back(MyVec::with_capacity(100));
        });

        Self {
            // available_vecs: 0,
            bufs,
        }
    }

    /// TODO
    pub fn get(&mut self) -> MyVec<T> {
        if self.bufs.is_empty() {
            self.bufs.push_back(MyVec::new());
        }

        self.bufs.pop_front().unwrap()
    }
}

/// TODO
pub fn clear_hashmap<K, T>(map: &mut HashMap<K, MyVec<T>>, vec_store: &mut VecStore<T>) {
    // println!("{}", map.len());
    // Move all Vec<T> out of the map and push them into vec_store
    for (_, mut v) in map.drain() {
        // println!("{}", v.len());
        v.clear();
        vec_store.bufs.push_back(v);
    }
}
