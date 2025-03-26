use std::collections::BTreeMap;

use dora_node_api::{ArrowData, Metadata, Parameter};
use kornia::image::{Image, ImageSize};

pub fn image_to_arrow(
    image: Image<u8, 3>,
    metadata: Metadata,
) -> eyre::Result<(BTreeMap<String, Parameter>, Vec<u8>)> {
    let mut meta_parameters = metadata.parameters;
    meta_parameters.insert("cols".to_string(), Parameter::Integer(image.cols() as i64));
    meta_parameters.insert("rows".to_string(), Parameter::Integer(image.rows() as i64));

    // TODO: avoid to_vec copy
    let data = image.as_slice().to_vec();

    Ok((meta_parameters, data))
}

pub fn arrow_to_image(data: ArrowData, metadata: Metadata) -> eyre::Result<Image<u8, 3>> {
    // SAFETY: we know that the metadata has the "cols" parameter
    let img_cols = metadata.parameters.get("cols").unwrap();
    let img_cols: i64 = match img_cols {
        Parameter::Integer(i) => *i,
        _ => return Err(eyre::eyre!("cols is not an integer")),
    };

    // SAFETY: we know that the metadata has the "rows" parameter
    let img_rows = metadata.parameters.get("rows").unwrap();
    let img_rows: i64 = match img_rows {
        Parameter::Integer(i) => *i,
        _ => return Err(eyre::eyre!("rows is not an integer")),
    };

    let img_data: Vec<u8> = TryFrom::try_from(&data)?;

    Ok(Image::new(
        ImageSize {
            width: img_cols as usize,
            height: img_rows as usize,
        },
        img_data,
    )?)
}
