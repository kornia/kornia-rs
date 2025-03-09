use std::collections::BTreeMap;

use arrow::array::PrimitiveArray;
use arrow::datatypes::UInt8Type;
use dora_node_api::{ArrowData, IntoArrow, Metadata, Parameter};
use kornia::image::{Image, ImageSize};

pub fn image_to_arrow(
    image: Image<u8, 3>,
    metadata: Metadata,
) -> eyre::Result<(BTreeMap<String, Parameter>, PrimitiveArray<UInt8Type>)> {
    let mut meta_parameters = metadata.parameters;
    meta_parameters.insert("cols".to_string(), Parameter::Integer(image.cols() as i64));
    meta_parameters.insert("rows".to_string(), Parameter::Integer(image.rows() as i64));

    // TODO: avoid to_vec copy
    let data = image.as_slice().to_vec().into_arrow();

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

// fn image_to_union_array(image: Image<u8, 3>) -> eyre::Result<UnionArray> {
//     let mut builder = UnionBuilder::new_dense();
//     builder.append::<Int32Type>("cols", image.cols() as i32)?;
//     builder.append::<Int32Type>("rows", image.rows() as i32)?;
//     // TODO: this is not efficient
//     for pixel in image.as_slice().iter() {
//         builder.append::<UInt8Type>("data", *pixel)?;
//     }
//     let union_array = builder.build()?;
//     Ok(union_array)
// }

// fn union_array_to_image(arrow: &arrow::array::UnionArray) -> eyre::Result<Image<u8, 3>> {
//     let cols = arrow
//         .value(0)
//         .as_any()
//         .downcast_ref::<arrow::array::Int32Array>()
//         .unwrap()
//         .value(0);
//     let rows = arrow
//         .value(1)
//         .as_any()
//         .downcast_ref::<arrow::array::Int32Array>()
//         .unwrap()
//         .value(0);
//     println!("cols: {}, rows: {}", cols, rows);
//
//     let mut data = Vec::with_capacity(cols as usize * rows as usize * 3);
//     for i in 0..(cols as usize * rows as usize * 3) {
//         data.push(
//             arrow
//                 .value(2 + i)
//                 .as_any()
//                 .downcast_ref::<arrow::array::UInt8Array>()
//                 .unwrap()
//                 .value(0),
//         );
//     }
//     println!("cols: {}, rows: {} len: {}", cols, rows, data.len());
//     let image = Image::new([cols as usize, rows as usize].into(), data)?;
//     Ok(image)
// }
