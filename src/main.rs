#![feature(portable_simd)]

use std::simd::f64x4;
use rand::Rng;
use rand::prelude::SliceRandom;
use std::time::Instant;


const CHUNK_SIZE: usize = 32; // 256 bits / 8 bits por byte

fn float_to_bits(value: f64) -> u8 {
    let mut rng = rand::thread_rng();
    let constant: f64 = rng.gen_range(0.0..=1.0);

    let fraction = value.fract();
    let randomized_fraction = (fraction + constant).fract();
    if randomized_fraction > 0.5 {
        1
    } else {
        0
    }
}


fn lorenz(x: f64x4, y: f64x4, z: f64x4, sigma: f64x4, rho: f64x4, beta: f64x4) -> (f64x4, f64x4, f64x4) {
    let dx = sigma * (y - x);
    let dy = x * (rho - z) - y;
    let dz = x * y - beta * z;
    (dx, dy, dz)
}

fn dopri<F>(f: F, x0: f64, y0: f64, z0: f64, t0: f64, tf: f64, dt: f64, _tol: f64) -> Vec<u8>
where
    F: Fn(f64x4, f64x4, f64x4, f64x4) -> (f64x4, f64x4, f64x4),
{
    let mut key = Vec::with_capacity(CHUNK_SIZE);

    let mut t = f64x4::splat(t0);
    let mut x = f64x4::splat(x0);
    let mut y = f64x4::splat(y0);
    let mut z = f64x4::splat(z0);
    let h = f64x4::splat(dt);

    // Constantes DOPRI como vetores SIMD
    let k = [
        f64x4::splat(0.2),
        f64x4::splat(0.3),
        f64x4::splat(0.075),
        f64x4::splat(0.225),
        f64x4::splat(0.8),
        f64x4::splat(0.8888888888888888),
        f64x4::splat(3.7333333333333334),
        f64x4::splat(3.5555555555555554),
        f64x4::splat(2.9525986892242035),
        f64x4::splat(11.595793324188385),
        f64x4::splat(9.822892851699436),
        f64x4::splat(0.2908093278463649),
        f64x4::splat(2.8462752525252526),
        f64x4::splat(10.757575757575758),
        f64x4::splat(8.906422717743473),
        f64x4::splat(0.2784090909090909),
        f64x4::splat(0.2735313036020583),
        f64x4::splat(0.09114583333333333),
        f64x4::splat(0.44923629829290207),
        f64x4::splat(0.6510416666666666),
        f64x4::splat(0.322376179245283),
        f64x4::splat(0.13095238095238096),
        f64x4::splat(0.025),
    ];

    let mut bit_accumulator: u8 = 0;
    let mut bit_count = 0;

    while t[0] < tf {
        if key.len() >= CHUNK_SIZE {
            break;
        }

        let t_k = [
            t + k[0] * h,
            t + k[1] * h,
            t + k[2] * h,
            t + k[3] * h,
            t + h,
        ];

        let (k1x, k1y, k1z) = f(x, y, z, t);
        let (k2x, k2y, k2z) = f(
            x + k[0] * h * k1x,
            y + k[0] * h * k1y,
            z + k[0] * h * k1z,
            t_k[0],
        );
        let (k3x, k3y, k3z) = f(
            x + k[2] * h * k1x + k[3] * h * k2x,
            y + k[2] * h * k1y + k[3] * h * k2y,
            z + k[2] * h * k1z + k[3] * h * k2z,
            t_k[1],
        );
        let (k4x, k4y, k4z) = f(
            x + k[6] * h * k1x - k[7] * h * k2x + k[8] * h * k3x,
            y + k[6] * h * k1y - k[7] * h * k2y + k[8] * h * k3y,
            z + k[6] * h * k1z - k[7] * h * k2z + k[8] * h * k3z,
            t_k[2],
        );
        let (k5x, k5y, k5z) = f(
            x + k[9] * h * k1x - k[10] * h * k2x + k[11] * h * k3x - k[12] * h * k4x,
            y + k[9] * h * k1y - k[10] * h * k2y + k[11] * h * k3y - k[12] * h * k4y,
            z + k[9] * h * k1z - k[10] * h * k2z + k[11] * h * k3z - k[12] * h * k4z,
            t_k[3],
        );
        let (k6x, k6y, k6z) = f(
            x + k[13] * h * k1x - k[14] * h * k2x + k[15] * h * k3x + k[16] * h * k4x - k[17] * h * k5x,
            y + k[13] * h * k1y - k[14] * h * k2y + k[15] * h * k3y + k[16] * h * k4y - k[17] * h * k5y,
            z + k[13] * h * k1z - k[14] * h * k2z + k[15] * h * k3z + k[16] * h * k4z - k[17] * h * k5z,
            t_k[4],
        );

        let x_new = x + h * (k[18] * k1x + k[19] * k3x + k[20] * k4x + k[21] * k5x + k[22] * k6x);
        let y_new = y + h * (k[18] * k1y + k[19] * k3y + k[20] * k4y + k[21] * k5y + k[22] * k6y);
        let z_new = z + h * (k[18] * k1z + k[19] * k3z + k[20] * k4z + k[21] * k5z + k[22] * k6z);

        t += h;
        x = x_new;
        y = y_new;
        z = z_new;

        let bits = [
            float_to_bits(x[0]),
            float_to_bits(y[0]),
            float_to_bits(z[0]),
            float_to_bits(x[1]),
            float_to_bits(y[1]),
            float_to_bits(z[1]),
            float_to_bits(x[2]),
            float_to_bits(y[2]),
            float_to_bits(z[2]),
            float_to_bits(x[3]),
            float_to_bits(y[3]),
            float_to_bits(z[3]),
        ];

        for &bit in &bits {
            bit_accumulator |= bit << bit_count;
            bit_count += 1;

            if bit_count >= 8 {
                key.push(bit_accumulator);
                bit_accumulator = 0;
                bit_count = 0;

                if key.len() >= CHUNK_SIZE {
                    break;
                }
            }
        }
    }
    key.truncate(CHUNK_SIZE);

    key
}

fn generate_chaotic_key() -> Vec<u8> {
    let mut rng = rand::thread_rng();

    let sigma = f64x4::splat(9.276605602496714);
    let rho = f64x4::splat(25.13288336025654);
    let beta = f64x4::splat(3.763256844113997);

    /*
    alguns valores utilizados antes para sigma, rho e beta.
    5.105006325607131
    44.27748459387101
    4.935246952078124
    */

    let x = rng.gen_range(0.0..30.0);
    let y = rng.gen_range(0.0..30.0);
    let z = rng.gen_range(0.0..30.0);

    let mut key = dopri(
        |x, y, z, _t| lorenz(x, y, z, sigma, rho, beta),
        x, y, z, 0.0, 2.6, 0.01, 1e-8
    );

    key.shuffle(&mut rng);

    key
}

fn generate_random_iv(size: usize) -> Vec<u8> {
    let mut iv = vec![0u8; size];
    rand::thread_rng().fill(&mut iv[..]);
    iv
}

fn modify_key_with_iv(key: &[u8], iv: &[u8]) -> Vec<u8> {
    let mut modified_key = vec![0u8; key.len()];
    for (i, &byte) in key.iter().enumerate() {
        modified_key[i] = byte ^ iv[i % iv.len()];
    }
    modified_key
}

fn bytes_to_string(bytes: &[u8]) -> String {
    bytes.iter().map(|&b| b as char).collect()
}

const SBOX: [u8; 256] = [
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab,
    0x76, 0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4,
    0x72, 0xc0, 0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71,
    0xd8, 0x31, 0x15, 0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2,
    0xeb, 0x27, 0xb2, 0x75, 0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6,
    0xb3, 0x29, 0xe3, 0x2f, 0x84, 0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb,
    0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf, 0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45,
    0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8, 0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5,
    0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2, 0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44,
    0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73, 0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a,
    0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb, 0xe0, 0x32, 0x3a, 0x0a, 0x49,
    0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79, 0xe7, 0xc8, 0x37, 0x6d,
    0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08, 0xba, 0x78, 0x25,
    0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a, 0x70, 0x3e,
    0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e, 0xe1,
    0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf,
    0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb,
    0x16,
];

fn compress_key(key: &[u8]) -> [u8; 32] {
    let mut compressed = [0u8; 32];
    for i in 0..4 {
        let start = i * 8;
        let part = &key[start..start + 8];
        let part_u64 = u64::from_le_bytes(part.try_into().unwrap());
        compressed[start..start + 8].copy_from_slice(&part_u64.to_le_bytes());
    }
    compressed
}

fn derive_key(original_key: &[u8; 32], sbox: [u8; 256]) -> [u8; 32] {
    let mut new_key = [0u8; 32];
    for i in 0..32 {
        let sbox_value = sbox[original_key[i] as usize];
        new_key[i] = original_key[i] ^ sbox_value;
    }
    new_key
}

fn lagrange_interpolation(x: u8, points: &[(u8, u8)]) -> u8 {
    let mut result = 0u8;
    let n = points.len() as u8;

    for i in 0..n {
        let (xi, yi) = points[i as usize];
        let mut term = yi;

        for j in 0..n {
            if i != j {
                let (xj, _) = points[j as usize];
                term = term.wrapping_mul(x.wrapping_sub(xj)).wrapping_div(xi.wrapping_sub(xj));
            }
        }

        result = result.wrapping_add(term);
    }

    result
}

fn polynomial_value(key: &[u64; 4], byte: u8) -> u8 {
    let key_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            key.as_ptr() as *const u8,
            key.len() * std::mem::size_of::<u64>()
        )
    };

    let points = (0..8).map(|i| {
        let xi = i as u8;
        let yi = if byte & (1 << i) != 0 { 1 } else { 0 };
        (xi, yi)
    }).collect::<Vec<_>>();

    let x = (key_bytes.iter().map(|&b| b as u64).sum::<u64>() % 256) as u8;
    lagrange_interpolation(x, &points)
}

fn process_text(text: &[u8], key: &[u64; 4], sbox: [u8; 256]) -> Vec<u8> {
    let mut processed_text = Vec::new();
    let mut i = 0;

    while i < text.len() {
        let mut group = [0u8; 4];
        let group_len = (text.len() - i).min(4);
        group[..group_len].copy_from_slice(&text[i..i + group_len]);

        // Aplicar preenchimento se o grupo for menor que 4 bytes
        if group_len < 4 {
            group[group_len] = 0x80;
            for j in (group_len + 1)..4 {
                group[j] = 0x00; // Preencher o restante com 0x00
            }
        }

        // Processar o grupo
        let first_byte = group[0];
        let poly_value = polynomial_value(key, first_byte);
        for j in 0..4 {
            group[j] ^= poly_value;
        }

        for j in 0..4 {
            group[j] = sbox[group[j] as usize];
        }

        processed_text.extend_from_slice(&group);
        i += 4;
    }

    processed_text
}

fn xor_and_sbox(text: &str, key: &[u8], sbox: [u8; 256]) -> Vec<u8> {
    let text_bytes = text.as_bytes();
    let key_len = key.len();
    let mut result = Vec::with_capacity(text_bytes.len());

    for (i, &byte) in text_bytes.iter().enumerate() {
        let key_byte = key[i % key_len];
        let xor_result = byte ^ key_byte;
        let sbox_result = sbox[xor_result as usize];
        result.push(sbox_result);
    }

    result
}

fn inv_process_text(text: &[u8], key: &[u64; 4], sbox_inverse: [u8; 256]) -> Vec<u8> {
    let mut processed_text = Vec::new();
    let mut i = 0;

    while i < text.len() {
        let mut group = [0u8; 4];
        let group_len = (text.len() - i).min(4);
        group[..group_len].copy_from_slice(&text[i..i + group_len]);

        // Aplicar a S-box inversa
        for j in 0..group_len {
            group[j] = sbox_inverse[group[j] as usize];
        }

        // Calcular o valor do polinômio para o primeiro byte do grupo
        let first_byte = group[0];
        let poly_value = polynomial_value(key, first_byte);

        // Aplicar XOR com o valor do polinômio
        for j in 0..group_len {
            group[j] ^= poly_value;
        }

        // Adicionar o grupo processado ao texto final
        processed_text.extend_from_slice(&group[..group_len]);

        i += 4;
    }

    // Remover o preenchimento (0x80) e bytes de preenchimento (0x00)
    if !processed_text.is_empty() {
        let mut trim_index = processed_text.len();
        while trim_index > 0 && processed_text[trim_index - 1] == 0x00 {
            trim_index -= 1;
        }
        // Verifica se o byte final é 0x80 e o remove se necessário
        if trim_index > 0 && processed_text[trim_index - 1] == 0x80 {
            trim_index -= 1;
        }
        processed_text.truncate(trim_index);
    }

    processed_text
}

fn inv_xor_and_sbox(encrypted_text: &[u8], key: &[u8], sbox_inverse: [u8; 256]) -> Vec<u8> {
    let key_len = key.len();
    let mut result = Vec::with_capacity(encrypted_text.len());

    for (i, &byte) in encrypted_text.iter().enumerate() {
        let key_byte = key[i % key_len];
        let sbox_inverse_result = sbox_inverse[byte as usize];
        let original_byte = sbox_inverse_result ^ key_byte;
        
        result.push(original_byte);
    }
    
    result
}

fn invert_sbox() -> [u8; 256] {
    let mut inv_sbox = [0u8; 256];
    for (i, &value) in SBOX.iter().enumerate() {
        inv_sbox[value as usize] = i as u8;
    }
    inv_sbox
}

fn extensive_processing(key: &[u8; 8]) -> u64 {
    let mut result = u64::from_le_bytes(*key);
    let constant = 0xA5A5A5A5A5A5A5A5;

    result = result.rotate_left(8) ^ constant;
    result = result.wrapping_add(0x12345678ABCDEF01 + u64::from(key[0]));
    result = result.rotate_right(16) ^ (constant ^ u64::from(key[1]));
    result = result.rotate_left(24);

    result
}

fn encrypt_text(text: &str, key: &[u8; 32]) -> Vec<u8> {
    let compressed_key = compress_key(key);
    let derived_key = derive_key(&compressed_key, SBOX);

    let mut ext_numbers = [0u64; 4];
    for i in 0..4 {
        let start = i * 8;
        let block: [u8; 8] = derived_key[start..start + 8].try_into().unwrap();
        ext_numbers[i] = extensive_processing(&block);
    }

    let processed_text = xor_and_sbox(text, &derived_key, SBOX);
    let final_number: &[u64; 4] = &ext_numbers;
    let result = process_text(&processed_text, final_number, SBOX);

    result
}

fn decrypt_text(encrypted_text: &[u8], key: &[u8; 32]) -> String {
    let compressed_key = compress_key(key);
    let derived_key = derive_key(&compressed_key, SBOX);

    let mut ext_numbers = [0u64; 4];
    for i in 0..4 {
        let start = i * 8;
        let block: [u8; 8] = derived_key[start..start + 8].try_into().unwrap();
        ext_numbers[i] = extensive_processing(&block);
    }

    let inv_sbox = invert_sbox();
    let processed_text = inv_process_text(encrypted_text, &ext_numbers, inv_sbox);

    let decrypted_text = inv_xor_and_sbox(&processed_text, &derived_key, inv_sbox);

    String::from_utf8(decrypted_text).unwrap_or_default()
}

fn main() {
    let start = Instant::now();

    let mut key = generate_chaotic_key();
    let mut rng = rand::thread_rng();
    key.shuffle(&mut rng);
    let iv = generate_random_iv(16);
    key = modify_key_with_iv(&key, &iv);
    key.shuffle(&mut rng);

    let text = "hello world";

    let key = compress_key(&key);
    let r = encrypt_text(text, &key);

    let mut l = String::new();

    l = decrypt_text(&r, &key);

    let dur = start.elapsed();
    println!("Execution time: {:.2?}", dur);

    println!("Key: \t {}", bytes_to_string(&key));
    let base64 = base64::encode(&key);
    println!("Key base64: {}", base64);

    println!("{}", bytes_to_string(&r));
    println!("{}", l);
}
