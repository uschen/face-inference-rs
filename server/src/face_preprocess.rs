use opencv::{core, prelude::*, Result};

pub fn mean_axis0(src: &core::Mat) -> Result<core::Mat> {
    let num = src.rows();
    let dim = src.cols();

    let mut output = unsafe { core::Mat::new_rows_cols(1, dim, core::CV_32F) }?;
    for i in 0..dim {
        let mut sum = 0_f32;
        for j in 0..num {
            sum += src.at_2d::<f32>(j, i)?;
        }
        *output.at_2d_mut::<f32>(0, i)? = sum / num as f32;
    }
    Ok(output)
}

pub fn elementwise_minus(a: &core::Mat, b: &core::Mat) -> Result<core::Mat> {
    assert!(b.cols() == a.cols());

    let mut output = unsafe { core::Mat::new_rows_cols(a.rows(), a.cols(), a.typ()?) }?;
    if b.cols() == a.cols() {
        for i in 0..a.rows() {
            for j in 0..b.cols() {
                *output.at_2d_mut::<f32>(i, j)? = *a.at_2d::<f32>(i, j)? - *b.at_2d::<f32>(0, j)?;
            }
        }
    }
    Ok(output)
}

pub fn var_axis0(src: &core::Mat) -> Result<core::Mat> {
    let temp = elementwise_minus(src, &mean_axis0(src)?)?;
    let mut output = temp.clone();
    core::multiply(&temp, &temp, &mut output, 1.0, -1)?;
    mean_axis0(&output)
}

pub fn matrix_rank(m: &core::Mat) -> Result<i32> {
    let mut w = core::Mat::default()?;
    let mut u = core::Mat::default()?;
    let mut vt = core::Mat::default()?;
    core::SVD::compute_ext(&m, &mut w, &mut u, &mut vt, 0)?;
    // core::Mat1b::
    let mut non_zero_singular_values = core::Mat::default()?;
    core::compare(
        &w,
        &core::Scalar::all(0.0001),
        &mut non_zero_singular_values,
        core::CMP_GT,
    )?;
    let rank = core::count_non_zero(&non_zero_singular_values)?;
    Ok(rank)
}

pub fn similar_transform(src: &core::Mat, dst: &core::Mat) -> Result<core::Mat> {
    let num = src.rows();
    let dim = src.cols();
    let src_mean = mean_axis0(src)?;
    let dst_mean = mean_axis0(dst)?;
    let src_demean = elementwise_minus(src, &src_mean)?;
    let dst_demean = elementwise_minus(dst, &dst_mean)?;
    let a_mat = core::div_matexpr_f64(
        &core::mul_matexpr_mat(&dst_demean.t()?, &src_demean)?,
        num as f64,
    )?
    .to_mat()?;

    let mut d_mat = unsafe { core::Mat::new_rows_cols(dim, 1, core::CV_32F) }?;
    d_mat.set_to(&core::Scalar::all(1.0), &core::no_array()?)?;
    if core::determinant(&a_mat)? < 0.0 {
        *d_mat.at_2d_mut::<f32>(dim - 1, 0)? = -1_f32;
    }

    let t_mat = core::Mat::eye(dim + 1, dim + 1, core::CV_32F)?.to_mat()?;
    let mut u_mat = core::Mat::default()?;
    let mut s_mat = core::Mat::default()?;
    let mut v_mat = core::Mat::default()?;
    core::SVD::compute_ext(&a_mat, &mut s_mat, &mut u_mat, &mut v_mat, 0)?;

    // the SVD function in opencv differ from scipy .

    let rank = matrix_rank(&a_mat)?;
    if rank == 0 {
        assert!(rank == 0);
    } else if rank == dim - 1 {
        if core::determinant(&u_mat)? * core::determinant(&v_mat)? > 0.0 {
            core::mul_mat_mat(&u_mat, &v_mat)?.to_mat()?.copy_to(
                &mut t_mat
                    .row_range(&core::Range::new(0, dim)?)?
                    .col_range(&core::Range::new(0, dim)?)?,
            )?;
        } else {
            //
            *d_mat.at_2d_mut::<f32>(dim - 1, 0)? = -1.0;
            let s = -1_f32;

            core::mul_mat_mat(&u_mat, &v_mat)?.to_mat()?.copy_to(
                &mut t_mat
                    .row_range(&core::Range::new(0, dim)?)?
                    .col_range(&core::Range::new(0, dim)?)?,
            )?;
            let diag = core::Mat::diag_mat(&d_mat)?;
            let twp = core::mul_mat_mat(&diag, &v_mat)?.to_mat()?;

            core::mul_mat_mat(&u_mat, &twp)?.to_mat()?.copy_to(
                &mut t_mat
                    .row_range(&core::Range::new(0, dim)?)?
                    .col_range(&core::Range::new(0, dim)?)?,
            )?;
            *d_mat.at_2d_mut::<f32>(dim - 1, 0)? = s;
        }
    } else {
        let diag = core::Mat::diag_mat(&d_mat)?;
        let twp = core::mul_mat_matexpr(&diag, &v_mat.t()?)?;
        // let res = core::mul_mat_matexpr(&u, &twp)?.to_mat()?;
        core::mul_matexpr_matexpr(&core::mul_f64_matexpr(-1.0, &u_mat.t()?)?, &twp)?
            .to_mat()?
            .copy_to(
                &mut t_mat
                    .row_range(&core::Range::new(0, dim)?)?
                    .col_range(&core::Range::new(0, dim)?)?,
            )?;
    }

    let var = var_axis0(&src_demean)?;
    let val = core::sum_elems(&var)?[0];
    let mut res = core::Mat::default()?;
    core::multiply(&d_mat, &s_mat, &mut res, 1.0, -1)?;
    let scale = 1.0 / val * core::sum_elems(&res)?[0];
    core::mul_f64_matexpr(
        -1.0,
        &t_mat
            .row_range(&core::Range::new(0, dim)?)?
            .col_range(&core::Range::new(0, dim)?)?
            .t()?,
    )?
    .to_mat()?
    .copy_to(
        &mut t_mat
            .row_range(&core::Range::new(0, dim)?)?
            .col_range(&core::Range::new(0, dim)?)?,
    )?;
    let temp1 = t_mat
        .row_range(&core::Range::new(0, dim)?)?
        .col_range(&core::Range::new(0, dim)?)?;
    let temp2 = src_mean.t()?.to_mat()?;
    let temp3 = core::mul_mat_mat(&temp1, &temp2)?;
    let temp4 = core::mul_f64_matexpr(scale, &temp3)?;
    core::mul_f64_matexpr(-1.0, &core::sub_matexpr_matexpr(&temp4, &dst_mean.t()?)?)?
        .to_mat()?
        .copy_to(
            &mut t_mat
                .row_range(&core::Range::new(0, dim)?)?
                .col_range(&core::Range::new(dim, dim + 1)?)?,
        )?;
    core::mul_f64_mat(
        scale,
        &t_mat
            .row_range(&core::Range::new(0, dim)?)?
            .col_range(&core::Range::new(0, dim)?)?,
    )?
    .to_mat()?
    .copy_to(
        &mut t_mat
            .row_range(&core::Range::new(0, dim)?)?
            .col_range(&core::Range::new(0, dim)?)?,
    )?;
    Ok(t_mat)
}

#[cfg(test)]
mod tests {
    use crate::face_preprocess;
    use opencv::{core, prelude::*};

    #[test]
    fn similar_transform() {
        let v1 = [
            [30.2946_f32, 51.6963_f32],
            [65.5318_f32, 51.5014_f32],
            [48.0252_f32, 71.7366_f32],
            [33.5493_f32, 92.3655_f32],
            [62.7299_f32, 92.2041_f32],
        ];
        let src = core::Mat::from_slice_2d(&v1).unwrap();

        let v2 = [
            [479.47498_f32, 224.70937_f32],
            [575.625_f32, 165.78749_f32],
            [579.45_f32, 213.39377_f32],
            [570.0562_f32, 306.97498_f32],
            [639.26245_f32, 260.73752_f32],
        ];

        let dst = core::Mat::from_slice_2d(&v2).unwrap();

        let m = face_preprocess::similar_transform(&dst, &src).unwrap();
        let exp = [
            [0.26081076_f32, -0.19798242_f32, -53.92479_f32].to_vec(),
            [0.19798243_f32, 0.26081076_f32, -101.81978_f32].to_vec(),
            [0.0_f32, 0.0_f32, 1.0_f32].to_vec(),
        ]
        .to_vec();
        assert_eq!(exp, m.to_vec_2d::<f32>().unwrap());
    }
}
