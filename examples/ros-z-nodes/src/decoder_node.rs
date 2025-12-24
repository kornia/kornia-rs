//! JPEG Decoder Node - subscribes to compressed JPEG images and publishes raw images

use crate::protos::{CompressedImage, Header, RawImage};
use kornia_image::{allocator::CpuAllocator, Image};
use kornia_io::jpeg::{decode_image_jpeg_layout, decode_image_jpeg_rgb8};
use prost::Message;
use ros_z::{
    context::ZContext, msg::ProtobufSerdes, node::ZNode, pubsub::ZSub, Builder, Result as ZResult,
};
use std::sync::Arc;
use zenoh::{
    bytes::ZBytes,
    pubsub::Publisher,
    sample::Sample,
    shm::{BlockOn, GarbageCollect, ShmProviderBuilder},
    Wait,
};

/// SHM pool size (256MB)
const SHM_POOL_SIZE: usize = 256 * 1024 * 1024;

fn get_pub_time() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0)
}

/// JPEG Decoder node - subscribes to compressed JPEG and publishes raw images via SHM
pub struct DecoderNode {
    #[allow(dead_code)]
    node: ZNode,
    subscriber: ZSub<CompressedImage, Sample, ProtobufSerdes<CompressedImage>>,
    camera_name: String,
    sequence: u32,
    shm_provider: Arc<zenoh::shm::ShmProvider<zenoh::shm::PosixShmProviderBackend>>,
    #[allow(dead_code)]
    shm_session: Arc<zenoh::Session>,
    shm_raw_pub: Publisher<'static>,
}

impl DecoderNode {
    /// Create a new JPEG decoder node
    pub fn new(ctx: Arc<ZContext>, camera_name: &str) -> ZResult<Self> {
        let node = ctx.create_node("decoder_node").build()?;

        let topic = format!("camera/{camera_name}/compressed");
        let subscriber = node
            .create_sub::<CompressedImage>(topic.as_str())
            .with_serdes::<ProtobufSerdes<CompressedImage>>()
            .build()?;

        log::info!("Decoder subscribing to '{}'", topic);

        let shm_provider = Arc::new(ShmProviderBuilder::default_backend(SHM_POOL_SIZE).wait()?);

        log::info!("Decoder SHM pool: {} MB", SHM_POOL_SIZE / (1024 * 1024));

        let mut zenoh_config = zenoh::Config::default();
        zenoh_config.insert_json5("transport/shared_memory/enabled", "true")?;

        let shm_session = Arc::new(zenoh::open(zenoh_config).wait()?);

        let shm_raw_topic = format!("camera/{camera_name}/raw_shm");
        log::info!("Decoder SHM topic: '{}'", shm_raw_topic);

        let shm_raw_pub = shm_session.declare_publisher(shm_raw_topic).wait()?;

        Ok(Self {
            node,
            subscriber,
            camera_name: camera_name.to_string(),
            shm_provider,
            shm_session,
            shm_raw_pub,
            sequence: 0,
        })
    }

    fn decode_jpeg(&mut self, msg: &CompressedImage) -> ZResult<RawImage> {
        let layout = decode_image_jpeg_layout(&msg.data)?;
        assert_eq!(layout.channels, 3);

        let mut image =
            Image::<u8, 3, CpuAllocator>::from_size_val(layout.image_size, 0, CpuAllocator)?;
        decode_image_jpeg_rgb8(&msg.data, &mut image)?;

        let sequence = self.sequence;
        self.sequence += 1;

        Ok(RawImage {
            header: Some(Header {
                acq_time: msg.header.as_ref().unwrap().acq_time,
                pub_time: get_pub_time(),
                sequence,
                frame_id: self.camera_name.clone(),
            }),
            width: layout.image_size.width as u32,
            height: layout.image_size.height as u32,
            encoding: "rgb8".to_string(),
            step: layout.image_size.width as u32 * 3,
            data: image.into_vec(),
        })
    }

    async fn publish_shm(&self, raw_image: RawImage) -> ZResult<()> {
        let proto_bytes = raw_image.encode_to_vec();

        let mut shm_buf = self
            .shm_provider
            .alloc(proto_bytes.len())
            .with_policy::<BlockOn<GarbageCollect>>()
            .await?;

        shm_buf.as_mut().copy_from_slice(&proto_bytes);
        let zbytes: ZBytes = shm_buf.into();

        self.shm_raw_pub.put(zbytes).await?;

        log::debug!(
            "[decoder] Frame {}: published ({}x{}, {} bytes)",
            raw_image.header.as_ref().unwrap().sequence,
            raw_image.width,
            raw_image.height,
            proto_bytes.len()
        );

        Ok(())
    }

    pub async fn run(mut self, shutdown_tx: tokio::sync::watch::Sender<()>) -> ZResult<()> {
        let mut shutdown_rx = shutdown_tx.subscribe();

        log::info!("Decoder node started for camera '{}'", self.camera_name);

        loop {
            tokio::select! {
                biased;

                _ = shutdown_rx.changed() => break,

                Ok(msg) = self.subscriber.async_recv() => {
                    if let Ok(raw_image) = self.decode_jpeg(&msg) {
                        if let Err(e) = self.publish_shm(raw_image).await {
                            log::error!("Failed to publish SHM buffer: {}", e);
                        }
                    }
                }
            }
        }

        Ok(())
    }
}
