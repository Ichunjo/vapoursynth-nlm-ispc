// based on KNLMeansCL by Khanattila and vs-boxblur

#define CLAMPX(x) clamp(x, 0, width - 1)
#define CLAMPY(y) clamp(y, 0, height - 1)

static inline uniform float square(uniform float x) {
    return x * x;
}

static inline float square(float x) {
    return x * x;
}

// positive difference
static inline float fdim(uniform float x, float y) {
    return (x > y) ? x - y : 0.0f;
}

static inline float welsch(float sum, uniform float h2_inv_norm) {
    return exp(-sum * h2_inv_norm);
}

static inline float bisquareA(float sum, uniform float h2_inv_norm) {
    return fdim(1.0f, sum * h2_inv_norm);
}

static inline float bisquareB(float sum, uniform float h2_inv_norm) {
    float tmp = fdim(1.0f, sum * h2_inv_norm);
    return tmp * tmp;
}

static inline float bisquareC(float sum, uniform float h2_inv_norm) {
    float tmp = fdim(1.0f, sum * h2_inv_norm);
    tmp *= tmp;
    tmp *= tmp;
    tmp *= tmp;
    return tmp;
}

/* =========================================================================
   Distance Kernels (Luma)
   ========================================================================= */

export void nlmDistanceLuma_f32(
    uniform float temp0[],
    uniform const float centerp[],
    uniform const float neighborp[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));

    for (uniform int y = 0; y < height; y++) {
        uniform int y_neighbor = CLAMPY(y + offset_y);
        uniform const float* uniform center_row = centerp + y * stride;
        uniform const float* uniform neighbor_row = neighborp + y_neighbor * stride;
        uniform float* uniform temp_row = temp0 + y * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform float u1 = center_row[x];
            uniform float u1_pq = neighbor_row[CLAMPX(x + offset_x)];
            temp_row[x] = 3.0f * square(u1 - u1_pq);
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                float u1 = center_row[x];
                float u1_pq = neighbor_row[x + offset_x];
                temp_row[x] = 3.0f * square(u1 - u1_pq);
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform float u1 = center_row[x];
            uniform float u1_pq = neighbor_row[CLAMPX(x + offset_x)];
            temp_row[x] = 3.0f * square(u1 - u1_pq);
        }
    }
}

export void nlmDistanceLuma_u8(
    uniform float temp0[],
    uniform const unsigned int8 centerp[],
    uniform const unsigned int8 neighborp[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride,
    uniform float inv_divisor
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));
    uniform float factor = 3.0f * square(inv_divisor);

    for (uniform int y = 0; y < height; y++) {
        uniform int y_neighbor = CLAMPY(y + offset_y);
        uniform const unsigned int8* uniform center_row = centerp + y * stride;
        uniform const unsigned int8* uniform neighbor_row = neighborp + y_neighbor * stride;
        uniform float* uniform temp_row = temp0 + y * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform int32 diff = (int32)center_row[x] - (int32)neighbor_row[CLAMPX(x + offset_x)];
            temp_row[x] = factor * (float)(diff * diff);
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                int32 diff = (int32)center_row[x] - (int32)neighbor_row[x + offset_x];
                temp_row[x] = factor * (float)(diff * diff);
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform int32 diff = (int32)center_row[x] - (int32)neighbor_row[CLAMPX(x + offset_x)];
            temp_row[x] = factor * (float)(diff * diff);
        }
    }
}

export void nlmDistanceLuma_u16(
    uniform float temp0[],
    uniform const unsigned int16 centerp[],
    uniform const unsigned int16 neighborp[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride,
    uniform float inv_divisor
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));
    uniform float factor = 3.0f * square(inv_divisor);

    for (uniform int y = 0; y < height; y++) {
        uniform int y_neighbor = CLAMPY(y + offset_y);
        uniform const unsigned int16* uniform center_row = centerp + y * stride;
        uniform const unsigned int16* uniform neighbor_row = neighborp + y_neighbor * stride;
        uniform float* uniform temp_row = temp0 + y * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform int32 diff = (int32)center_row[x] - (int32)neighbor_row[CLAMPX(x + offset_x)];
            temp_row[x] = factor * (float)(diff * diff);
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                int32 diff = (int32)center_row[x] - (int32)neighbor_row[x + offset_x];
                temp_row[x] = factor * (float)(diff * diff);
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform int32 diff = (int32)center_row[x] - (int32)neighbor_row[CLAMPX(x + offset_x)];
            temp_row[x] = factor * (float)(diff * diff);
        }
    }
}

/* =========================================================================
   Distance Kernels (Chroma)
   ========================================================================= */

export void nlmDistanceChroma_f32(
    uniform float temp0[],
    uniform const float centerp1[],
    uniform const float centerp2[],
    uniform const float neighborp1[],
    uniform const float neighborp2[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));

    for (uniform int y = 0; y < height; y++) {
        uniform int y_neighbor = CLAMPY(y + offset_y);
        uniform const float* uniform c_row1 = centerp1 + y * stride;
        uniform const float* uniform c_row2 = centerp2 + y * stride;
        uniform const float* uniform n_row1 = neighborp1 + y_neighbor * stride;
        uniform const float* uniform n_row2 = neighborp2 + y_neighbor * stride;
        uniform float* uniform temp_row = temp0 + y * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform int n_x = CLAMPX(x + offset_x);
            uniform float u1_1 = c_row1[x];
            uniform float u1_2 = c_row2[x];
            uniform float u1_pq_1 = n_row1[n_x];
            uniform float u1_pq_2 = n_row2[n_x];
            temp_row[x] = 1.5f * (square(u1_1 - u1_pq_1) + square(u1_2 - u1_pq_2));
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                float u1_1 = c_row1[x];
                float u1_2 = c_row2[x];
                float u1_pq_1 = n_row1[x + offset_x];
                float u1_pq_2 = n_row2[x + offset_x];
                temp_row[x] = 1.5f * (square(u1_1 - u1_pq_1) + square(u1_2 - u1_pq_2));
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform int n_x = CLAMPX(x + offset_x);
            uniform float u1_1 = c_row1[x];
            uniform float u1_2 = c_row2[x];
            uniform float u1_pq_1 = n_row1[n_x];
            uniform float u1_pq_2 = n_row2[n_x];
            temp_row[x] = 1.5f * (square(u1_1 - u1_pq_1) + square(u1_2 - u1_pq_2));
        }
    }
}

export void nlmDistanceChroma_u8(
    uniform float temp0[],
    uniform const unsigned int8 centerp1[],
    uniform const unsigned int8 centerp2[],
    uniform const unsigned int8 neighborp1[],
    uniform const unsigned int8 neighborp2[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride,
    uniform float inv_divisor
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));
    uniform float factor = 1.5f * square(inv_divisor);

    for (uniform int y = 0; y < height; y++) {
        uniform int y_neighbor = CLAMPY(y + offset_y);
        uniform const unsigned int8* uniform c_row1 = centerp1 + y * stride;
        uniform const unsigned int8* uniform c_row2 = centerp2 + y * stride;
        uniform const unsigned int8* uniform n_row1 = neighborp1 + y_neighbor * stride;
        uniform const unsigned int8* uniform n_row2 = neighborp2 + y_neighbor * stride;
        uniform float* uniform temp_row = temp0 + y * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform int n_x = CLAMPX(x + offset_x);
            uniform int32 d1 = (int32)c_row1[x] - (int32)n_row1[n_x];
            uniform int32 d2 = (int32)c_row2[x] - (int32)n_row2[n_x];
            temp_row[x] = factor * (float)(d1 * d1 + d2 * d2);
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                int32 d1 = (int32)c_row1[x] - (int32)n_row1[x + offset_x];
                int32 d2 = (int32)c_row2[x] - (int32)n_row2[x + offset_x];
                temp_row[x] = factor * (float)(d1 * d1 + d2 * d2);
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform int n_x = CLAMPX(x + offset_x);
            uniform int32 d1 = (int32)c_row1[x] - (int32)n_row1[n_x];
            uniform int32 d2 = (int32)c_row2[x] - (int32)n_row2[n_x];
            temp_row[x] = factor * (float)(d1 * d1 + d2 * d2);
        }
    }
}

export void nlmDistanceChroma_u16(
    uniform float temp0[],
    uniform const unsigned int16 centerp1[],
    uniform const unsigned int16 centerp2[],
    uniform const unsigned int16 neighborp1[],
    uniform const unsigned int16 neighborp2[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride,
    uniform float inv_divisor
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));
    uniform float factor = 1.5f * square(inv_divisor);

    for (uniform int y = 0; y < height; y++) {
        uniform int y_neighbor = CLAMPY(y + offset_y);
        uniform const unsigned int16* uniform c_row1 = centerp1 + y * stride;
        uniform const unsigned int16* uniform c_row2 = centerp2 + y * stride;
        uniform const unsigned int16* uniform n_row1 = neighborp1 + y_neighbor * stride;
        uniform const unsigned int16* uniform n_row2 = neighborp2 + y_neighbor * stride;
        uniform float* uniform temp_row = temp0 + y * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform int n_x = CLAMPX(x + offset_x);
            uniform int32 d1 = (int32)c_row1[x] - (int32)n_row1[n_x];
            uniform int32 d2 = (int32)c_row2[x] - (int32)n_row2[n_x];
            temp_row[x] = factor * (float)(d1 * d1 + d2 * d2);
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                int32 d1 = (int32)c_row1[x] - (int32)n_row1[x + offset_x];
                int32 d2 = (int32)c_row2[x] - (int32)n_row2[x + offset_x];
                temp_row[x] = factor * (float)(d1 * d1 + d2 * d2);
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform int n_x = CLAMPX(x + offset_x);
            uniform int32 d1 = (int32)c_row1[x] - (int32)n_row1[n_x];
            uniform int32 d2 = (int32)c_row2[x] - (int32)n_row2[n_x];
            temp_row[x] = factor * (float)(d1 * d1 + d2 * d2);
        }
    }
}

/* =========================================================================
   Distance Kernels (YUV)
   ========================================================================= */

export void nlmDistanceYUV_f32(
    uniform float temp0[],
    uniform const float centerp1[],
    uniform const float centerp2[],
    uniform const float centerp3[],
    uniform const float neighborp1[],
    uniform const float neighborp2[],
    uniform const float neighborp3[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));

    for (uniform int y = 0; y < height; y++) {
        uniform int y_neighbor = CLAMPY(y + offset_y);
        uniform const float* uniform c_row1 = centerp1 + y * stride;
        uniform const float* uniform c_row2 = centerp2 + y * stride;
        uniform const float* uniform c_row3 = centerp3 + y * stride;
        uniform const float* uniform n_row1 = neighborp1 + y_neighbor * stride;
        uniform const float* uniform n_row2 = neighborp2 + y_neighbor * stride;
        uniform const float* uniform n_row3 = neighborp3 + y_neighbor * stride;
        uniform float* uniform temp_row = temp0 + y * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform int n_x = CLAMPX(x + offset_x);
            uniform float u1_1 = c_row1[x];
            uniform float u1_2 = c_row2[x];
            uniform float u1_3 = c_row3[x];
            uniform float u1_pq_1 = n_row1[n_x];
            uniform float u1_pq_2 = n_row2[n_x];
            uniform float u1_pq_3 = n_row3[n_x];
            temp_row[x] = square(u1_1 - u1_pq_1) + square(u1_2 - u1_pq_2) + square(u1_3 - u1_pq_3);
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                float u1_1 = c_row1[x];
                float u1_2 = c_row2[x];
                float u1_3 = c_row3[x];
                float u1_pq_1 = n_row1[x + offset_x];
                float u1_pq_2 = n_row2[x + offset_x];
                float u1_pq_3 = n_row3[x + offset_x];
                temp_row[x] = square(u1_1 - u1_pq_1) + square(u1_2 - u1_pq_2) + square(u1_3 - u1_pq_3);
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform int n_x = CLAMPX(x + offset_x);
            uniform float u1_1 = c_row1[x];
            uniform float u1_2 = c_row2[x];
            uniform float u1_3 = c_row3[x];
            uniform float u1_pq_1 = n_row1[n_x];
            uniform float u1_pq_2 = n_row2[n_x];
            uniform float u1_pq_3 = n_row3[n_x];
            temp_row[x] = square(u1_1 - u1_pq_1) + square(u1_2 - u1_pq_2) + square(u1_3 - u1_pq_3);
        }
    }
}

export void nlmDistanceYUV_u8(
    uniform float temp0[],
    uniform const unsigned int8 centerp1[],
    uniform const unsigned int8 centerp2[],
    uniform const unsigned int8 centerp3[],
    uniform const unsigned int8 neighborp1[],
    uniform const unsigned int8 neighborp2[],
    uniform const unsigned int8 neighborp3[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride,
    uniform float inv_divisor
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));
    uniform float factor = square(inv_divisor);

    for (uniform int y = 0; y < height; y++) {
        uniform int y_neighbor = CLAMPY(y + offset_y);
        uniform const unsigned int8* uniform c_row1 = centerp1 + y * stride;
        uniform const unsigned int8* uniform c_row2 = centerp2 + y * stride;
        uniform const unsigned int8* uniform c_row3 = centerp3 + y * stride;
        uniform const unsigned int8* uniform n_row1 = neighborp1 + y_neighbor * stride;
        uniform const unsigned int8* uniform n_row2 = neighborp2 + y_neighbor * stride;
        uniform const unsigned int8* uniform n_row3 = neighborp3 + y_neighbor * stride;
        uniform float* uniform temp_row = temp0 + y * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform int n_x = CLAMPX(x + offset_x);
            uniform int32 d1 = (int32)c_row1[x] - (int32)n_row1[n_x];
            uniform int32 d2 = (int32)c_row2[x] - (int32)n_row2[n_x];
            uniform int32 d3 = (int32)c_row3[x] - (int32)n_row3[n_x];
            temp_row[x] = factor * (float)(d1 * d1 + d2 * d2 + d3 * d3);
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                int32 d1 = (int32)c_row1[x] - (int32)n_row1[x + offset_x];
                int32 d2 = (int32)c_row2[x] - (int32)n_row2[x + offset_x];
                int32 d3 = (int32)c_row3[x] - (int32)n_row3[x + offset_x];
                temp_row[x] = factor * (float)(d1 * d1 + d2 * d2 + d3 * d3);
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform int n_x = CLAMPX(x + offset_x);
            uniform int32 d1 = (int32)c_row1[x] - (int32)n_row1[n_x];
            uniform int32 d2 = (int32)c_row2[x] - (int32)n_row2[n_x];
            uniform int32 d3 = (int32)c_row3[x] - (int32)n_row3[n_x];
            temp_row[x] = factor * (float)(d1 * d1 + d2 * d2 + d3 * d3);
        }
    }
}

export void nlmDistanceYUV_u16(
    uniform float temp0[],
    uniform const unsigned int16 centerp1[],
    uniform const unsigned int16 centerp2[],
    uniform const unsigned int16 centerp3[],
    uniform const unsigned int16 neighborp1[],
    uniform const unsigned int16 neighborp2[],
    uniform const unsigned int16 neighborp3[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride,
    uniform float inv_divisor
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));
    uniform float factor = square(inv_divisor);

    for (uniform int y = 0; y < height; y++) {
        uniform int y_neighbor = CLAMPY(y + offset_y);
        uniform const unsigned int16* uniform c_row1 = centerp1 + y * stride;
        uniform const unsigned int16* uniform c_row2 = centerp2 + y * stride;
        uniform const unsigned int16* uniform c_row3 = centerp3 + y * stride;
        uniform const unsigned int16* uniform n_row1 = neighborp1 + y_neighbor * stride;
        uniform const unsigned int16* uniform n_row2 = neighborp2 + y_neighbor * stride;
        uniform const unsigned int16* uniform n_row3 = neighborp3 + y_neighbor * stride;
        uniform float* uniform temp_row = temp0 + y * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform int n_x = CLAMPX(x + offset_x);
            uniform int32 d1 = (int32)c_row1[x] - (int32)n_row1[n_x];
            uniform int32 d2 = (int32)c_row2[x] - (int32)n_row2[n_x];
            uniform int32 d3 = (int32)c_row3[x] - (int32)n_row3[n_x];
            temp_row[x] = factor * (float)(d1 * d1 + d2 * d2 + d3 * d3);
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                int32 d1 = (int32)c_row1[x] - (int32)n_row1[x + offset_x];
                int32 d2 = (int32)c_row2[x] - (int32)n_row2[x + offset_x];
                int32 d3 = (int32)c_row3[x] - (int32)n_row3[x + offset_x];
                temp_row[x] = factor * (float)(d1 * d1 + d2 * d2 + d3 * d3);
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform int n_x = CLAMPX(x + offset_x);
            uniform int32 d1 = (int32)c_row1[x] - (int32)n_row1[n_x];
            uniform int32 d2 = (int32)c_row2[x] - (int32)n_row2[n_x];
            uniform int32 d3 = (int32)c_row3[x] - (int32)n_row3[n_x];
            temp_row[x] = factor * (float)(d1 * d1 + d2 * d2 + d3 * d3);
        }
    }
}

/* =========================================================================
   Distance Kernels (RGB)
   ========================================================================= */

export void nlmDistanceRGB_f32(
    uniform float temp0[],
    uniform const float centerp1[],
    uniform const float centerp2[],
    uniform const float centerp3[],
    uniform const float neighborp1[],
    uniform const float neighborp2[],
    uniform const float neighborp3[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));

    for (uniform int y = 0; y < height; y++) {
        uniform int y_neighbor = CLAMPY(y + offset_y);
        uniform const float* uniform c_row1 = centerp1 + y * stride;
        uniform const float* uniform c_row2 = centerp2 + y * stride;
        uniform const float* uniform c_row3 = centerp3 + y * stride;
        uniform const float* uniform n_row1 = neighborp1 + y_neighbor * stride;
        uniform const float* uniform n_row2 = neighborp2 + y_neighbor * stride;
        uniform const float* uniform n_row3 = neighborp3 + y_neighbor * stride;
        uniform float* uniform temp_row = temp0 + y * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform int n_x = CLAMPX(x + offset_x);
            uniform float u1_1 = c_row1[x];
            uniform float u1_2 = c_row2[x];
            uniform float u1_3 = c_row3[x];
            uniform float u1_pq_1 = n_row1[n_x];
            uniform float u1_pq_2 = n_row2[n_x];
            uniform float u1_pq_3 = n_row3[n_x];
            uniform float m_red = (u1_1 + u1_pq_1) / 6.0f;
            temp_row[x] = (2.0f / 3.0f + m_red) * square(u1_1 - u1_pq_1) + (4.0f / 3.0f) * square(u1_2 - u1_pq_2) +
                          (1.0f - m_red) * square(u1_3 - u1_pq_3);
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                float u1_1 = c_row1[x];
                float u1_2 = c_row2[x];
                float u1_3 = c_row3[x];
                float u1_pq_1 = n_row1[x + offset_x];
                float u1_pq_2 = n_row2[x + offset_x];
                float u1_pq_3 = n_row3[x + offset_x];
                float m_red = (u1_1 + u1_pq_1) / 6.0f;
                temp_row[x] = (2.0f / 3.0f + m_red) * square(u1_1 - u1_pq_1) + (4.0f / 3.0f) * square(u1_2 - u1_pq_2) +
                              (1.0f - m_red) * square(u1_3 - u1_pq_3);
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform int n_x = CLAMPX(x + offset_x);
            uniform float u1_1 = c_row1[x];
            uniform float u1_2 = c_row2[x];
            uniform float u1_3 = c_row3[x];
            uniform float u1_pq_1 = n_row1[n_x];
            uniform float u1_pq_2 = n_row2[n_x];
            uniform float u1_pq_3 = n_row3[n_x];
            uniform float m_red = (u1_1 + u1_pq_1) / 6.0f;
            temp_row[x] = (2.0f / 3.0f + m_red) * square(u1_1 - u1_pq_1) + (4.0f / 3.0f) * square(u1_2 - u1_pq_2) +
                          (1.0f - m_red) * square(u1_3 - u1_pq_3);
        }
    }
}

export void nlmDistanceRGB_u8(
    uniform float temp0[],
    uniform const unsigned int8 centerp1[],
    uniform const unsigned int8 centerp2[],
    uniform const unsigned int8 centerp3[],
    uniform const unsigned int8 neighborp1[],
    uniform const unsigned int8 neighborp2[],
    uniform const unsigned int8 neighborp3[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride,
    uniform float inv_divisor
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));
    uniform float sq_inv_divisor = square(inv_divisor);
    uniform float inv_6_divisor = inv_divisor / 6.0f;

    for (uniform int y = 0; y < height; y++) {
        uniform int y_neighbor = CLAMPY(y + offset_y);
        uniform const unsigned int8* uniform c_row1 = centerp1 + y * stride;
        uniform const unsigned int8* uniform c_row2 = centerp2 + y * stride;
        uniform const unsigned int8* uniform c_row3 = centerp3 + y * stride;
        uniform const unsigned int8* uniform n_row1 = neighborp1 + y_neighbor * stride;
        uniform const unsigned int8* uniform n_row2 = neighborp2 + y_neighbor * stride;
        uniform const unsigned int8* uniform n_row3 = neighborp3 + y_neighbor * stride;
        uniform float* uniform temp_row = temp0 + y * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform int n_x = CLAMPX(x + offset_x);
            uniform float u1_1 = c_row1[x];
            uniform float u1_2 = c_row2[x];
            uniform float u1_3 = c_row3[x];
            uniform float u1_pq_1 = n_row1[n_x];
            uniform float u1_pq_2 = n_row2[n_x];
            uniform float u1_pq_3 = n_row3[n_x];
            uniform float m_red = (u1_1 + u1_pq_1) * inv_6_divisor;
            temp_row[x] = ((2.0f / 3.0f + m_red) * square(u1_1 - u1_pq_1) + (4.0f / 3.0f) * square(u1_2 - u1_pq_2) +
                           (1.0f - m_red) * square(u1_3 - u1_pq_3)) *
                          sq_inv_divisor;
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                float u1_1 = c_row1[x];
                float u1_2 = c_row2[x];
                float u1_3 = c_row3[x];
                float u1_pq_1 = n_row1[x + offset_x];
                float u1_pq_2 = n_row2[x + offset_x];
                float u1_pq_3 = n_row3[x + offset_x];
                float m_red = (u1_1 + u1_pq_1) * inv_6_divisor;
                temp_row[x] = ((2.0f / 3.0f + m_red) * square(u1_1 - u1_pq_1) + (4.0f / 3.0f) * square(u1_2 - u1_pq_2) +
                               (1.0f - m_red) * square(u1_3 - u1_pq_3)) *
                              sq_inv_divisor;
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform int n_x = CLAMPX(x + offset_x);
            uniform float u1_1 = c_row1[x];
            uniform float u1_2 = c_row2[x];
            uniform float u1_3 = c_row3[x];
            uniform float u1_pq_1 = n_row1[n_x];
            uniform float u1_pq_2 = n_row2[n_x];
            uniform float u1_pq_3 = n_row3[n_x];
            uniform float m_red = (u1_1 + u1_pq_1) * inv_6_divisor;
            temp_row[x] = ((2.0f / 3.0f + m_red) * square(u1_1 - u1_pq_1) + (4.0f / 3.0f) * square(u1_2 - u1_pq_2) +
                           (1.0f - m_red) * square(u1_3 - u1_pq_3)) *
                          sq_inv_divisor;
        }
    }
}

export void nlmDistanceRGB_u16(
    uniform float temp0[],
    uniform const unsigned int16 centerp1[],
    uniform const unsigned int16 centerp2[],
    uniform const unsigned int16 centerp3[],
    uniform const unsigned int16 neighborp1[],
    uniform const unsigned int16 neighborp2[],
    uniform const unsigned int16 neighborp3[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride,
    uniform float inv_divisor
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));
    uniform float sq_inv_divisor = square(inv_divisor);
    uniform float inv_6_divisor = inv_divisor / 6.0f;

    for (uniform int y = 0; y < height; y++) {
        uniform int y_neighbor = CLAMPY(y + offset_y);
        uniform const unsigned int16* uniform c_row1 = centerp1 + y * stride;
        uniform const unsigned int16* uniform c_row2 = centerp2 + y * stride;
        uniform const unsigned int16* uniform c_row3 = centerp3 + y * stride;
        uniform const unsigned int16* uniform n_row1 = neighborp1 + y_neighbor * stride;
        uniform const unsigned int16* uniform n_row2 = neighborp2 + y_neighbor * stride;
        uniform const unsigned int16* uniform n_row3 = neighborp3 + y_neighbor * stride;
        uniform float* uniform temp_row = temp0 + y * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform int n_x = CLAMPX(x + offset_x);
            uniform float u1_1 = c_row1[x];
            uniform float u1_2 = c_row2[x];
            uniform float u1_3 = c_row3[x];
            uniform float u1_pq_1 = n_row1[n_x];
            uniform float u1_pq_2 = n_row2[n_x];
            uniform float u1_pq_3 = n_row3[n_x];
            uniform float m_red = (u1_1 + u1_pq_1) * inv_6_divisor;
            temp_row[x] = ((2.0f / 3.0f + m_red) * square(u1_1 - u1_pq_1) + (4.0f / 3.0f) * square(u1_2 - u1_pq_2) +
                           (1.0f - m_red) * square(u1_3 - u1_pq_3)) *
                          sq_inv_divisor;
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                float u1_1 = c_row1[x];
                float u1_2 = c_row2[x];
                float u1_3 = c_row3[x];
                float u1_pq_1 = n_row1[x + offset_x];
                float u1_pq_2 = n_row2[x + offset_x];
                float u1_pq_3 = n_row3[x + offset_x];
                float m_red = (u1_1 + u1_pq_1) * inv_6_divisor;
                temp_row[x] = ((2.0f / 3.0f + m_red) * square(u1_1 - u1_pq_1) + (4.0f / 3.0f) * square(u1_2 - u1_pq_2) +
                               (1.0f - m_red) * square(u1_3 - u1_pq_3)) *
                              sq_inv_divisor;
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform int n_x = CLAMPX(x + offset_x);
            uniform float u1_1 = c_row1[x];
            uniform float u1_2 = c_row2[x];
            uniform float u1_3 = c_row3[x];
            uniform float u1_pq_1 = n_row1[n_x];
            uniform float u1_pq_2 = n_row2[n_x];
            uniform float u1_pq_3 = n_row3[n_x];
            uniform float m_red = (u1_1 + u1_pq_1) * inv_6_divisor;
            temp_row[x] = ((2.0f / 3.0f + m_red) * square(u1_1 - u1_pq_1) + (4.0f / 3.0f) * square(u1_2 - u1_pq_2) +
                           (1.0f - m_red) * square(u1_3 - u1_pq_3)) *
                          sq_inv_divisor;
        }
    }
}

/* =========================================================================
   Horizontal Box Blur Filter
   ========================================================================= */

static void nlmHorizontalS0(
    uniform float temp0[],
    uniform const float temp[],
    uniform int width,
    uniform int height,
    uniform int stride
) {
    for (uniform int y = 0; y < height; y++) {
        uniform const float* uniform src_row = temp + y * stride;
        uniform float* uniform dst_row = temp0 + y * stride;
        foreach (x = 0 ... width) {
            dst_row[x] = src_row[x];
        }
    }
}

static void nlmHorizontalS1(
    uniform float temp0[],
    uniform const float temp[],
    uniform int width,
    uniform int height,
    uniform int stride
) {
    uniform int end = width - 1;

    for (uniform int y = 0; y < height; y++) {
        uniform const float* uniform src_row = temp + y * stride;
        uniform float* uniform dst_row = temp0 + y * stride;

        dst_row[0] = src_row[0] * 2.0f + src_row[1];

        foreach (x = 1 ... end) {
            dst_row[x] = src_row[x - 1] + src_row[x] + src_row[x + 1];
        }

        if (end > 0) {
            dst_row[width - 1] = src_row[width - 2] + src_row[width - 1] * 2.0f;
        }
    }
}

static void nlmHorizontalS2(
    uniform float temp0[],
    uniform const float temp[],
    uniform int width,
    uniform int height,
    uniform int stride
) {
    uniform int end = width - 2;

    for (uniform int y = 0; y < height; y++) {
        uniform const float* uniform src_row = temp + y * stride;
        uniform float* uniform dst_row = temp0 + y * stride;

        for (uniform int x = 0; x < min(2, width); x++) {
            uniform float sum = 0.0f;
#pragma unroll
            for (uniform int j = -2; j <= 2; ++j) {
                sum += src_row[CLAMPX(x + j)];
            }
            dst_row[x] = sum;
        }

        if (end > 2) {
            foreach (x = 2 ... end) {
                dst_row[x] = (src_row[x - 2] + src_row[x - 1]) + (src_row[x] + (src_row[x + 1] + src_row[x + 2]));
            }
        }

        for (uniform int x = max(2, end); x < width; x++) {
            uniform float sum = 0.0f;
#pragma unroll
            for (uniform int j = -2; j <= 2; ++j) {
                sum += src_row[CLAMPX(x + j)];
            }
            dst_row[x] = sum;
        }
    }
}

static void nlmHorizontalS3(
    uniform float temp0[],
    uniform const float temp[],
    uniform int width,
    uniform int height,
    uniform int stride
) {
    uniform int end = width - 3;

    for (uniform int y = 0; y < height; y++) {
        uniform const float* uniform src_row = temp + y * stride;
        uniform float* uniform dst_row = temp0 + y * stride;

        for (uniform int x = 0; x < min(3, width); x++) {
            uniform float sum = 0.0f;
#pragma unroll
            for (uniform int j = -3; j <= 3; ++j) {
                sum += src_row[CLAMPX(x + j)];
            }
            dst_row[x] = sum;
        }

        if (end > 3) {
            foreach (x = 3 ... end) {
                dst_row[x] = ((src_row[x - 3] + src_row[x - 2]) + (src_row[x - 1] + src_row[x])) +
                             ((src_row[x + 1] + src_row[x + 2]) + src_row[x + 3]);
            }
        }

        for (uniform int x = max(3, end); x < width; x++) {
            uniform float sum = 0.0f;
#pragma unroll
            for (uniform int j = -3; j <= 3; ++j) {
                sum += src_row[CLAMPX(x + j)];
            }
            dst_row[x] = sum;
        }
    }
}

static void nlmHorizontalS4(
    uniform float temp0[],
    uniform const float temp[],
    uniform int width,
    uniform int height,
    uniform int stride
) {
    uniform int end = width - 4;

    for (uniform int y = 0; y < height; y++) {
        uniform const float* uniform src_row = temp + y * stride;
        uniform float* uniform dst_row = temp0 + y * stride;

        for (uniform int x = 0; x < min(4, width); x++) {
            uniform float sum = 0.0f;
#pragma unroll
            for (uniform int j = -4; j <= 4; ++j) {
                sum += src_row[CLAMPX(x + j)];
            }
            dst_row[x] = sum;
        }

        if (end > 4) {
            foreach (x = 4 ... end) {
                dst_row[x] = (((src_row[x - 4] + src_row[x - 3]) + (src_row[x - 2] + src_row[x - 1])) + src_row[x]) +
                             (((src_row[x + 1] + src_row[x + 2]) + src_row[x + 3]) + src_row[x + 4]);
            }
        }

        for (uniform int x = max(4, end); x < width; x++) {
            uniform float sum = 0.0f;
#pragma unroll
            for (uniform int j = -4; j <= 4; ++j) {
                sum += src_row[CLAMPX(x + j)];
            }
            dst_row[x] = sum;
        }
    }
}

export void nlmHorizontal(
    uniform float temp0[],
    uniform const float temp[],
    uniform int nlm_s,
    uniform int width,
    uniform int height,
    uniform int stride
) {
    if (nlm_s == 0) {
        nlmHorizontalS0(temp0, temp, width, height, stride);
        return;
    } else if (nlm_s == 1) {
        nlmHorizontalS1(temp0, temp, width, height, stride);
        return;
    } else if (nlm_s == 2) {
        nlmHorizontalS2(temp0, temp, width, height, stride);
        return;
    } else if (nlm_s == 3) {
        nlmHorizontalS3(temp0, temp, width, height, stride);
        return;
    } else if (nlm_s == 4) {
        nlmHorizontalS4(temp0, temp, width, height, stride);
        return;
    }

    uniform int start = nlm_s;
    uniform int end = width - nlm_s;

    for (uniform int y = 0; y < height; y++) {
        uniform const float* uniform src_row = temp + y * stride;
        uniform float* uniform dst_row = temp0 + y * stride;

        for (uniform int x = 0; x < min(start, width); x++) {
            uniform float sum = 0.0f;
            for (uniform int j = -nlm_s; j <= nlm_s; ++j) {
                sum += src_row[CLAMPX(x + j)];
            }
            dst_row[x] = sum;
        }

        if (end > start) {
            foreach (x = start... end) {
                float sum = 0.0f;
                for (uniform int j = -nlm_s; j <= nlm_s; ++j) {
                    sum += src_row[x + j];
                }
                dst_row[x] = sum;
            }
        }

        for (uniform int x = max(start, end); x < width; x++) {
            uniform float sum = 0.0f;
            for (uniform int j = -nlm_s; j <= nlm_s; ++j) {
                sum += src_row[CLAMPX(x + j)];
            }
            dst_row[x] = sum;
        }
    }
}

/* =========================================================================
   Vertical Box Blur & Weight Kernels
   ========================================================================= */

export void nlmVerticalWelsch(
    uniform float dstp[],
    uniform const float srcp[],
    uniform int radius,
    uniform float h2_inv_norm,
    uniform int width,
    uniform int height,
    uniform int stride,
    uniform float buffer[]
) {
    foreach (x = 0 ... width) {
        buffer[x] = radius * srcp[x];
    }

    for (uniform int y = 0; y < radius; ++y) {
        uniform const float* uniform row = srcp + min(y, height - 1) * stride;
        foreach (x = 0 ... width) {
            buffer[x] += row[x];
        }
    }

    for (uniform int y = 0; y < min(radius, height); ++y) {
        uniform const float* uniform add_row = srcp + min(y + radius, height - 1) * stride;
        uniform const float* uniform sub_row = srcp;
        uniform float* uniform dst_row = dstp + y * stride;
        foreach (x = 0 ... width) {
            float b = buffer[x] + add_row[x];
            dst_row[x] = welsch(b, h2_inv_norm);
            buffer[x] = b - sub_row[x];
        }
    }

    if (height > radius) {
        for (uniform int y = radius; y < height - radius; ++y) {
            uniform const float* uniform add_row = srcp + (y + radius) * stride;
            uniform const float* uniform sub_row = srcp + (y - radius) * stride;
            uniform float* uniform dst_row = dstp + y * stride;
            foreach (x = 0 ... width) {
                float b = buffer[x] + add_row[x];
                dst_row[x] = welsch(b, h2_inv_norm);
                buffer[x] = b - sub_row[x];
            }
        }

        for (uniform int y = max(height - radius, radius); y < height; ++y) {
            uniform const float* uniform add_row = srcp + min(y + radius, height - 1) * stride;
            uniform const float* uniform sub_row = srcp + (y - radius) * stride;
            uniform float* uniform dst_row = dstp + y * stride;
            foreach (x = 0 ... width) {
                float b = buffer[x] + add_row[x];
                dst_row[x] = welsch(b, h2_inv_norm);
                buffer[x] = b - sub_row[x];
            }
        }
    }
}

export void nlmVerticalBisquareA(
    uniform float dstp[],
    uniform const float srcp[],
    uniform int radius,
    uniform float h2_inv_norm,
    uniform int width,
    uniform int height,
    uniform int stride,
    uniform float buffer[]
) {
    foreach (x = 0 ... width) {
        buffer[x] = radius * srcp[x];
    }

    for (uniform int y = 0; y < radius; ++y) {
        uniform const float* uniform row = srcp + min(y, height - 1) * stride;
        foreach (x = 0 ... width) {
            buffer[x] += row[x];
        }
    }

    for (uniform int y = 0; y < min(radius, height); ++y) {
        uniform const float* uniform add_row = srcp + min(y + radius, height - 1) * stride;
        uniform const float* uniform sub_row = srcp;
        uniform float* uniform dst_row = dstp + y * stride;
        foreach (x = 0 ... width) {
            float b = buffer[x] + add_row[x];
            dst_row[x] = bisquareA(b, h2_inv_norm);
            buffer[x] = b - sub_row[x];
        }
    }

    if (height > radius) {
        for (uniform int y = radius; y < height - radius; ++y) {
            uniform const float* uniform add_row = srcp + (y + radius) * stride;
            uniform const float* uniform sub_row = srcp + (y - radius) * stride;
            uniform float* uniform dst_row = dstp + y * stride;
            foreach (x = 0 ... width) {
                float b = buffer[x] + add_row[x];
                dst_row[x] = bisquareA(b, h2_inv_norm);
                buffer[x] = b - sub_row[x];
            }
        }

        for (uniform int y = max(height - radius, radius); y < height; ++y) {
            uniform const float* uniform add_row = srcp + min(y + radius, height - 1) * stride;
            uniform const float* uniform sub_row = srcp + (y - radius) * stride;
            uniform float* uniform dst_row = dstp + y * stride;
            foreach (x = 0 ... width) {
                float b = buffer[x] + add_row[x];
                dst_row[x] = bisquareA(b, h2_inv_norm);
                buffer[x] = b - sub_row[x];
            }
        }
    }
}

export void nlmVerticalBisquareB(
    uniform float dstp[],
    uniform const float srcp[],
    uniform int radius,
    uniform float h2_inv_norm,
    uniform int width,
    uniform int height,
    uniform int stride,
    uniform float buffer[]
) {
    foreach (x = 0 ... width) {
        buffer[x] = radius * srcp[x];
    }

    for (uniform int y = 0; y < radius; ++y) {
        uniform const float* uniform row = srcp + min(y, height - 1) * stride;
        foreach (x = 0 ... width) {
            buffer[x] += row[x];
        }
    }

    for (uniform int y = 0; y < min(radius, height); ++y) {
        uniform const float* uniform add_row = srcp + min(y + radius, height - 1) * stride;
        uniform const float* uniform sub_row = srcp;
        uniform float* uniform dst_row = dstp + y * stride;
        foreach (x = 0 ... width) {
            float b = buffer[x] + add_row[x];
            dst_row[x] = bisquareB(b, h2_inv_norm);
            buffer[x] = b - sub_row[x];
        }
    }

    if (height > radius) {
        for (uniform int y = radius; y < height - radius; ++y) {
            uniform const float* uniform add_row = srcp + (y + radius) * stride;
            uniform const float* uniform sub_row = srcp + (y - radius) * stride;
            uniform float* uniform dst_row = dstp + y * stride;
            foreach (x = 0 ... width) {
                float b = buffer[x] + add_row[x];
                dst_row[x] = bisquareB(b, h2_inv_norm);
                buffer[x] = b - sub_row[x];
            }
        }

        for (uniform int y = max(height - radius, radius); y < height; ++y) {
            uniform const float* uniform add_row = srcp + min(y + radius, height - 1) * stride;
            uniform const float* uniform sub_row = srcp + (y - radius) * stride;
            uniform float* uniform dst_row = dstp + y * stride;
            foreach (x = 0 ... width) {
                float b = buffer[x] + add_row[x];
                dst_row[x] = bisquareB(b, h2_inv_norm);
                buffer[x] = b - sub_row[x];
            }
        }
    }
}

export void nlmVerticalBisquareC(
    uniform float dstp[],
    uniform const float srcp[],
    uniform int radius,
    uniform float h2_inv_norm,
    uniform int width,
    uniform int height,
    uniform int stride,
    uniform float buffer[]
) {
    foreach (x = 0 ... width) {
        buffer[x] = radius * srcp[x];
    }

    for (uniform int y = 0; y < radius; ++y) {
        uniform const float* uniform row = srcp + min(y, height - 1) * stride;
        foreach (x = 0 ... width) {
            buffer[x] += row[x];
        }
    }

    for (uniform int y = 0; y < min(radius, height); ++y) {
        uniform const float* uniform add_row = srcp + min(y + radius, height - 1) * stride;
        uniform const float* uniform sub_row = srcp;
        uniform float* uniform dst_row = dstp + y * stride;
        foreach (x = 0 ... width) {
            float b = buffer[x] + add_row[x];
            dst_row[x] = bisquareC(b, h2_inv_norm);
            buffer[x] = b - sub_row[x];
        }
    }

    if (height > radius) {
        for (uniform int y = radius; y < height - radius; ++y) {
            uniform const float* uniform add_row = srcp + (y + radius) * stride;
            uniform const float* uniform sub_row = srcp + (y - radius) * stride;
            uniform float* uniform dst_row = dstp + y * stride;
            foreach (x = 0 ... width) {
                float b = buffer[x] + add_row[x];
                dst_row[x] = bisquareC(b, h2_inv_norm);
                buffer[x] = b - sub_row[x];
            }
        }

        for (uniform int y = max(height - radius, radius); y < height; ++y) {
            uniform const float* uniform add_row = srcp + min(y + radius, height - 1) * stride;
            uniform const float* uniform sub_row = srcp + (y - radius) * stride;
            uniform float* uniform dst_row = dstp + y * stride;
            foreach (x = 0 ... width) {
                float b = buffer[x] + add_row[x];
                dst_row[x] = bisquareC(b, h2_inv_norm);
                buffer[x] = b - sub_row[x];
            }
        }
    }
}

/* =========================================================================
   Accumulation Kernels (1 Channel)
   ========================================================================= */

export void nlmAccumulationCh1_f32(
    uniform float weightp[],
    uniform float wdstp[],
    uniform float max_weightp[],
    uniform const float srcp_bwd[],
    uniform const float srcp_fwd[],
    uniform const float temp1[],
    uniform const float temp2[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));

    for (uniform int y = 0; y < height; y++) {
        uniform int y_mq = CLAMPY(y - offset_y);
        uniform int y_pq = CLAMPY(y + offset_y);

        uniform float* uniform weight_row = weightp + y * stride;
        uniform float* uniform max_weight_row = max_weightp + y * stride;
        uniform float* uniform wdst_row = wdstp + y * stride;

        uniform const float* uniform t1_row = temp1 + y * stride;
        uniform const float* uniform t2_row = temp2 + y_mq * stride;
        uniform const float* uniform src_bwd_row = srcp_bwd + y_pq * stride;
        uniform const float* uniform src_fwd_row = srcp_fwd + y_mq * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform float u4 = t1_row[x];
            uniform float u4_mq = t2_row[CLAMPX(x - offset_x)];
            weight_row[x] += u4 + u4_mq;
            max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
            uniform float u1_pq = src_bwd_row[CLAMPX(x + offset_x)];
            uniform float u1_mq = src_fwd_row[CLAMPX(x - offset_x)];
            wdst_row[x] += u4 * u1_pq + u4_mq * u1_mq;
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                float u4 = t1_row[x];
                float u4_mq = t2_row[x - offset_x];
                weight_row[x] += u4 + u4_mq;
                max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
                float u1_pq = src_bwd_row[x + offset_x];
                float u1_mq = src_fwd_row[x - offset_x];
                wdst_row[x] += u4 * u1_pq + u4_mq * u1_mq;
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform float u4 = t1_row[x];
            uniform float u4_mq = t2_row[CLAMPX(x - offset_x)];
            weight_row[x] += u4 + u4_mq;
            max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
            uniform float u1_pq = src_bwd_row[CLAMPX(x + offset_x)];
            uniform float u1_mq = src_fwd_row[CLAMPX(x - offset_x)];
            wdst_row[x] += u4 * u1_pq + u4_mq * u1_mq;
        }
    }
}

export void nlmAccumulationCh1_u8(
    uniform float weightp[],
    uniform float wdstp[],
    uniform float max_weightp[],
    uniform const unsigned int8 srcp_bwd[],
    uniform const unsigned int8 srcp_fwd[],
    uniform const float temp1[],
    uniform const float temp2[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));

    for (uniform int y = 0; y < height; y++) {
        uniform int y_mq = CLAMPY(y - offset_y);
        uniform int y_pq = CLAMPY(y + offset_y);

        uniform float* uniform weight_row = weightp + y * stride;
        uniform float* uniform max_weight_row = max_weightp + y * stride;
        uniform float* uniform wdst_row = wdstp + y * stride;

        uniform const float* uniform t1_row = temp1 + y * stride;
        uniform const float* uniform t2_row = temp2 + y_mq * stride;
        uniform const unsigned int8* uniform src_bwd_row = srcp_bwd + y_pq * stride;
        uniform const unsigned int8* uniform src_fwd_row = srcp_fwd + y_mq * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform float u4 = t1_row[x];
            uniform float u4_mq = t2_row[CLAMPX(x - offset_x)];
            weight_row[x] += u4 + u4_mq;
            max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
            uniform float u1_pq = src_bwd_row[CLAMPX(x + offset_x)];
            uniform float u1_mq = src_fwd_row[CLAMPX(x - offset_x)];
            wdst_row[x] += u4 * u1_pq + u4_mq * u1_mq;
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                float u4 = t1_row[x];
                float u4_mq = t2_row[x - offset_x];
                weight_row[x] += u4 + u4_mq;
                max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
                float u1_pq = src_bwd_row[x + offset_x];
                float u1_mq = src_fwd_row[x - offset_x];
                wdst_row[x] += u4 * u1_pq + u4_mq * u1_mq;
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform float u4 = t1_row[x];
            uniform float u4_mq = t2_row[CLAMPX(x - offset_x)];
            weight_row[x] += u4 + u4_mq;
            max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
            uniform float u1_pq = src_bwd_row[CLAMPX(x + offset_x)];
            uniform float u1_mq = src_fwd_row[CLAMPX(x - offset_x)];
            wdst_row[x] += u4 * u1_pq + u4_mq * u1_mq;
        }
    }
}

export void nlmAccumulationCh1_u16(
    uniform float weightp[],
    uniform float wdstp[],
    uniform float max_weightp[],
    uniform const unsigned int16 srcp_bwd[],
    uniform const unsigned int16 srcp_fwd[],
    uniform const float temp1[],
    uniform const float temp2[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));

    for (uniform int y = 0; y < height; y++) {
        uniform int y_mq = CLAMPY(y - offset_y);
        uniform int y_pq = CLAMPY(y + offset_y);

        uniform float* uniform weight_row = weightp + y * stride;
        uniform float* uniform max_weight_row = max_weightp + y * stride;
        uniform float* uniform wdst_row = wdstp + y * stride;

        uniform const float* uniform t1_row = temp1 + y * stride;
        uniform const float* uniform t2_row = temp2 + y_mq * stride;
        uniform const unsigned int16* uniform src_bwd_row = srcp_bwd + y_pq * stride;
        uniform const unsigned int16* uniform src_fwd_row = srcp_fwd + y_mq * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform float u4 = t1_row[x];
            uniform float u4_mq = t2_row[CLAMPX(x - offset_x)];
            weight_row[x] += u4 + u4_mq;
            max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
            uniform float u1_pq = src_bwd_row[CLAMPX(x + offset_x)];
            uniform float u1_mq = src_fwd_row[CLAMPX(x - offset_x)];
            wdst_row[x] += u4 * u1_pq + u4_mq * u1_mq;
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                float u4 = t1_row[x];
                float u4_mq = t2_row[x - offset_x];
                weight_row[x] += u4 + u4_mq;
                max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
                float u1_pq = src_bwd_row[x + offset_x];
                float u1_mq = src_fwd_row[x - offset_x];
                wdst_row[x] += u4 * u1_pq + u4_mq * u1_mq;
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform float u4 = t1_row[x];
            uniform float u4_mq = t2_row[CLAMPX(x - offset_x)];
            weight_row[x] += u4 + u4_mq;
            max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
            uniform float u1_pq = src_bwd_row[CLAMPX(x + offset_x)];
            uniform float u1_mq = src_fwd_row[CLAMPX(x - offset_x)];
            wdst_row[x] += u4 * u1_pq + u4_mq * u1_mq;
        }
    }
}

/* =========================================================================
   Accumulation Kernels (2 Channels)
   ========================================================================= */

export void nlmAccumulationCh2_f32(
    uniform float weightp[],
    uniform float wdstp1[],
    uniform float wdstp2[],
    uniform float max_weightp[],
    uniform const float srcp_bwd1[],
    uniform const float srcp_bwd2[],
    uniform const float srcp_fwd1[],
    uniform const float srcp_fwd2[],
    uniform const float temp1[],
    uniform const float temp2[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));

    for (uniform int y = 0; y < height; y++) {
        uniform int y_mq = CLAMPY(y - offset_y);
        uniform int y_pq = CLAMPY(y + offset_y);

        uniform float* uniform weight_row = weightp + y * stride;
        uniform float* uniform max_weight_row = max_weightp + y * stride;
        uniform float* uniform wdst_row1 = wdstp1 + y * stride;
        uniform float* uniform wdst_row2 = wdstp2 + y * stride;

        uniform const float* uniform t1_row = temp1 + y * stride;
        uniform const float* uniform t2_row = temp2 + y_mq * stride;
        uniform const float* uniform src_bwd_row1 = srcp_bwd1 + y_pq * stride;
        uniform const float* uniform src_bwd_row2 = srcp_bwd2 + y_pq * stride;
        uniform const float* uniform src_fwd_row1 = srcp_fwd1 + y_mq * stride;
        uniform const float* uniform src_fwd_row2 = srcp_fwd2 + y_mq * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform float u4 = t1_row[x];
            uniform float u4_mq = t2_row[CLAMPX(x - offset_x)];
            weight_row[x] += u4 + u4_mq;
            max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
            uniform int pq_x = CLAMPX(x + offset_x);
            uniform int mq_x = CLAMPX(x - offset_x);
            uniform float u1_pq_1 = src_bwd_row1[pq_x];
            uniform float u1_pq_2 = src_bwd_row2[pq_x];
            uniform float u1_mq_1 = src_fwd_row1[mq_x];
            uniform float u1_mq_2 = src_fwd_row2[mq_x];
            wdst_row1[x] += u4 * u1_pq_1 + u4_mq * u1_mq_1;
            wdst_row2[x] += u4 * u1_pq_2 + u4_mq * u1_mq_2;
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                float u4 = t1_row[x];
                float u4_mq = t2_row[x - offset_x];
                weight_row[x] += u4 + u4_mq;
                max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
                float u1_pq_1 = src_bwd_row1[x + offset_x];
                float u1_pq_2 = src_bwd_row2[x + offset_x];
                float u1_mq_1 = src_fwd_row1[x - offset_x];
                float u1_mq_2 = src_fwd_row2[x - offset_x];
                wdst_row1[x] += u4 * u1_pq_1 + u4_mq * u1_mq_1;
                wdst_row2[x] += u4 * u1_pq_2 + u4_mq * u1_mq_2;
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform float u4 = t1_row[x];
            uniform float u4_mq = t2_row[CLAMPX(x - offset_x)];
            weight_row[x] += u4 + u4_mq;
            max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
            uniform int pq_x = CLAMPX(x + offset_x);
            uniform int mq_x = CLAMPX(x - offset_x);
            uniform float u1_pq_1 = src_bwd_row1[pq_x];
            uniform float u1_pq_2 = src_bwd_row2[pq_x];
            uniform float u1_mq_1 = src_fwd_row1[mq_x];
            uniform float u1_mq_2 = src_fwd_row2[mq_x];
            wdst_row1[x] += u4 * u1_pq_1 + u4_mq * u1_mq_1;
            wdst_row2[x] += u4 * u1_pq_2 + u4_mq * u1_mq_2;
        }
    }
}

export void nlmAccumulationCh2_u8(
    uniform float weightp[],
    uniform float wdstp1[],
    uniform float wdstp2[],
    uniform float max_weightp[],
    uniform const unsigned int8 srcp_bwd1[],
    uniform const unsigned int8 srcp_bwd2[],
    uniform const unsigned int8 srcp_fwd1[],
    uniform const unsigned int8 srcp_fwd2[],
    uniform const float temp1[],
    uniform const float temp2[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));

    for (uniform int y = 0; y < height; y++) {
        uniform int y_mq = CLAMPY(y - offset_y);
        uniform int y_pq = CLAMPY(y + offset_y);

        uniform float* uniform weight_row = weightp + y * stride;
        uniform float* uniform max_weight_row = max_weightp + y * stride;
        uniform float* uniform wdst_row1 = wdstp1 + y * stride;
        uniform float* uniform wdst_row2 = wdstp2 + y * stride;

        uniform const float* uniform t1_row = temp1 + y * stride;
        uniform const float* uniform t2_row = temp2 + y_mq * stride;
        uniform const unsigned int8* uniform src_bwd_row1 = srcp_bwd1 + y_pq * stride;
        uniform const unsigned int8* uniform src_bwd_row2 = srcp_bwd2 + y_pq * stride;
        uniform const unsigned int8* uniform src_fwd_row1 = srcp_fwd1 + y_mq * stride;
        uniform const unsigned int8* uniform src_fwd_row2 = srcp_fwd2 + y_mq * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform float u4 = t1_row[x];
            uniform float u4_mq = t2_row[CLAMPX(x - offset_x)];
            weight_row[x] += u4 + u4_mq;
            max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
            uniform int pq_x = CLAMPX(x + offset_x);
            uniform int mq_x = CLAMPX(x - offset_x);
            uniform float u1_pq_1 = src_bwd_row1[pq_x];
            uniform float u1_pq_2 = src_bwd_row2[pq_x];
            uniform float u1_mq_1 = src_fwd_row1[mq_x];
            uniform float u1_mq_2 = src_fwd_row2[mq_x];
            wdst_row1[x] += u4 * u1_pq_1 + u4_mq * u1_mq_1;
            wdst_row2[x] += u4 * u1_pq_2 + u4_mq * u1_mq_2;
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                float u4 = t1_row[x];
                float u4_mq = t2_row[x - offset_x];
                weight_row[x] += u4 + u4_mq;
                max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
                float u1_pq_1 = src_bwd_row1[x + offset_x];
                float u1_pq_2 = src_bwd_row2[x + offset_x];
                float u1_mq_1 = src_fwd_row1[x - offset_x];
                float u1_mq_2 = src_fwd_row2[x - offset_x];
                wdst_row1[x] += u4 * u1_pq_1 + u4_mq * u1_mq_1;
                wdst_row2[x] += u4 * u1_pq_2 + u4_mq * u1_mq_2;
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform float u4 = t1_row[x];
            uniform float u4_mq = t2_row[CLAMPX(x - offset_x)];
            weight_row[x] += u4 + u4_mq;
            max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
            uniform int pq_x = CLAMPX(x + offset_x);
            uniform int mq_x = CLAMPX(x - offset_x);
            uniform float u1_pq_1 = src_bwd_row1[pq_x];
            uniform float u1_pq_2 = src_bwd_row2[pq_x];
            uniform float u1_mq_1 = src_fwd_row1[mq_x];
            uniform float u1_mq_2 = src_fwd_row2[mq_x];
            wdst_row1[x] += u4 * u1_pq_1 + u4_mq * u1_mq_1;
            wdst_row2[x] += u4 * u1_pq_2 + u4_mq * u1_mq_2;
        }
    }
}

export void nlmAccumulationCh2_u16(
    uniform float weightp[],
    uniform float wdstp1[],
    uniform float wdstp2[],
    uniform float max_weightp[],
    uniform const unsigned int16 srcp_bwd1[],
    uniform const unsigned int16 srcp_bwd2[],
    uniform const unsigned int16 srcp_fwd1[],
    uniform const unsigned int16 srcp_fwd2[],
    uniform const float temp1[],
    uniform const float temp2[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));

    for (uniform int y = 0; y < height; y++) {
        uniform int y_mq = CLAMPY(y - offset_y);
        uniform int y_pq = CLAMPY(y + offset_y);

        uniform float* uniform weight_row = weightp + y * stride;
        uniform float* uniform max_weight_row = max_weightp + y * stride;
        uniform float* uniform wdst_row1 = wdstp1 + y * stride;
        uniform float* uniform wdst_row2 = wdstp2 + y * stride;

        uniform const float* uniform t1_row = temp1 + y * stride;
        uniform const float* uniform t2_row = temp2 + y_mq * stride;
        uniform const unsigned int16* uniform src_bwd_row1 = srcp_bwd1 + y_pq * stride;
        uniform const unsigned int16* uniform src_bwd_row2 = srcp_bwd2 + y_pq * stride;
        uniform const unsigned int16* uniform src_fwd_row1 = srcp_fwd1 + y_mq * stride;
        uniform const unsigned int16* uniform src_fwd_row2 = srcp_fwd2 + y_mq * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform float u4 = t1_row[x];
            uniform float u4_mq = t2_row[CLAMPX(x - offset_x)];
            weight_row[x] += u4 + u4_mq;
            max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
            uniform int pq_x = CLAMPX(x + offset_x);
            uniform int mq_x = CLAMPX(x - offset_x);
            uniform float u1_pq_1 = src_bwd_row1[pq_x];
            uniform float u1_pq_2 = src_bwd_row2[pq_x];
            uniform float u1_mq_1 = src_fwd_row1[mq_x];
            uniform float u1_mq_2 = src_fwd_row2[mq_x];
            wdst_row1[x] += u4 * u1_pq_1 + u4_mq * u1_mq_1;
            wdst_row2[x] += u4 * u1_pq_2 + u4_mq * u1_mq_2;
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                float u4 = t1_row[x];
                float u4_mq = t2_row[x - offset_x];
                weight_row[x] += u4 + u4_mq;
                max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
                float u1_pq_1 = src_bwd_row1[x + offset_x];
                float u1_pq_2 = src_bwd_row2[x + offset_x];
                float u1_mq_1 = src_fwd_row1[x - offset_x];
                float u1_mq_2 = src_fwd_row2[x - offset_x];
                wdst_row1[x] += u4 * u1_pq_1 + u4_mq * u1_mq_1;
                wdst_row2[x] += u4 * u1_pq_2 + u4_mq * u1_mq_2;
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform float u4 = t1_row[x];
            uniform float u4_mq = t2_row[CLAMPX(x - offset_x)];
            weight_row[x] += u4 + u4_mq;
            max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
            uniform int pq_x = CLAMPX(x + offset_x);
            uniform int mq_x = CLAMPX(x - offset_x);
            uniform float u1_pq_1 = src_bwd_row1[pq_x];
            uniform float u1_pq_2 = src_bwd_row2[pq_x];
            uniform float u1_mq_1 = src_fwd_row1[mq_x];
            uniform float u1_mq_2 = src_fwd_row2[mq_x];
            wdst_row1[x] += u4 * u1_pq_1 + u4_mq * u1_mq_1;
            wdst_row2[x] += u4 * u1_pq_2 + u4_mq * u1_mq_2;
        }
    }
}

/* =========================================================================
   Accumulation Kernels (3 Channels)
   ========================================================================= */

export void nlmAccumulationCh3_f32(
    uniform float weightp[],
    uniform float wdstp1[],
    uniform float wdstp2[],
    uniform float wdstp3[],
    uniform float max_weightp[],
    uniform const float srcp_bwd1[],
    uniform const float srcp_bwd2[],
    uniform const float srcp_bwd3[],
    uniform const float srcp_fwd1[],
    uniform const float srcp_fwd2[],
    uniform const float srcp_fwd3[],
    uniform const float temp1[],
    uniform const float temp2[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));

    for (uniform int y = 0; y < height; y++) {
        uniform int y_mq = CLAMPY(y - offset_y);
        uniform int y_pq = CLAMPY(y + offset_y);

        uniform float* uniform weight_row = weightp + y * stride;
        uniform float* uniform max_weight_row = max_weightp + y * stride;
        uniform float* uniform wdst_row1 = wdstp1 + y * stride;
        uniform float* uniform wdst_row2 = wdstp2 + y * stride;
        uniform float* uniform wdst_row3 = wdstp3 + y * stride;

        uniform const float* uniform t1_row = temp1 + y * stride;
        uniform const float* uniform t2_row = temp2 + y_mq * stride;
        uniform const float* uniform src_bwd_row1 = srcp_bwd1 + y_pq * stride;
        uniform const float* uniform src_bwd_row2 = srcp_bwd2 + y_pq * stride;
        uniform const float* uniform src_bwd_row3 = srcp_bwd3 + y_pq * stride;
        uniform const float* uniform src_fwd_row1 = srcp_fwd1 + y_mq * stride;
        uniform const float* uniform src_fwd_row2 = srcp_fwd2 + y_mq * stride;
        uniform const float* uniform src_fwd_row3 = srcp_fwd3 + y_mq * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform float u4 = t1_row[x];
            uniform float u4_mq = t2_row[CLAMPX(x - offset_x)];
            weight_row[x] += u4 + u4_mq;
            max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
            uniform int pq_x = CLAMPX(x + offset_x);
            uniform int mq_x = CLAMPX(x - offset_x);
            uniform float u1_pq_1 = src_bwd_row1[pq_x];
            uniform float u1_pq_2 = src_bwd_row2[pq_x];
            uniform float u1_pq_3 = src_bwd_row3[pq_x];
            uniform float u1_mq_1 = src_fwd_row1[mq_x];
            uniform float u1_mq_2 = src_fwd_row2[mq_x];
            uniform float u1_mq_3 = src_fwd_row3[mq_x];
            wdst_row1[x] += u4 * u1_pq_1 + u4_mq * u1_mq_1;
            wdst_row2[x] += u4 * u1_pq_2 + u4_mq * u1_mq_2;
            wdst_row3[x] += u4 * u1_pq_3 + u4_mq * u1_mq_3;
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                float u4 = t1_row[x];
                float u4_mq = t2_row[x - offset_x];
                weight_row[x] += u4 + u4_mq;
                max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
                float u1_pq_1 = src_bwd_row1[x + offset_x];
                float u1_pq_2 = src_bwd_row2[x + offset_x];
                float u1_pq_3 = src_bwd_row3[x + offset_x];
                float u1_mq_1 = src_fwd_row1[x - offset_x];
                float u1_mq_2 = src_fwd_row2[x - offset_x];
                float u1_mq_3 = src_fwd_row3[x - offset_x];
                wdst_row1[x] += u4 * u1_pq_1 + u4_mq * u1_mq_1;
                wdst_row2[x] += u4 * u1_pq_2 + u4_mq * u1_mq_2;
                wdst_row3[x] += u4 * u1_pq_3 + u4_mq * u1_mq_3;
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform float u4 = t1_row[x];
            uniform float u4_mq = t2_row[CLAMPX(x - offset_x)];
            weight_row[x] += u4 + u4_mq;
            max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
            uniform int pq_x = CLAMPX(x + offset_x);
            uniform int mq_x = CLAMPX(x - offset_x);
            uniform float u1_pq_1 = src_bwd_row1[pq_x];
            uniform float u1_pq_2 = src_bwd_row2[pq_x];
            uniform float u1_pq_3 = src_bwd_row3[pq_x];
            uniform float u1_mq_1 = src_fwd_row1[mq_x];
            uniform float u1_mq_2 = src_fwd_row2[mq_x];
            uniform float u1_mq_3 = src_fwd_row3[mq_x];
            wdst_row1[x] += u4 * u1_pq_1 + u4_mq * u1_mq_1;
            wdst_row2[x] += u4 * u1_pq_2 + u4_mq * u1_mq_2;
            wdst_row3[x] += u4 * u1_pq_3 + u4_mq * u1_mq_3;
        }
    }
}

export void nlmAccumulationCh3_u8(
    uniform float weightp[],
    uniform float wdstp1[],
    uniform float wdstp2[],
    uniform float wdstp3[],
    uniform float max_weightp[],
    uniform const unsigned int8 srcp_bwd1[],
    uniform const unsigned int8 srcp_bwd2[],
    uniform const unsigned int8 srcp_bwd3[],
    uniform const unsigned int8 srcp_fwd1[],
    uniform const unsigned int8 srcp_fwd2[],
    uniform const unsigned int8 srcp_fwd3[],
    uniform const float temp1[],
    uniform const float temp2[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));

    for (uniform int y = 0; y < height; y++) {
        uniform int y_mq = CLAMPY(y - offset_y);
        uniform int y_pq = CLAMPY(y + offset_y);

        uniform float* uniform weight_row = weightp + y * stride;
        uniform float* uniform max_weight_row = max_weightp + y * stride;
        uniform float* uniform wdst_row1 = wdstp1 + y * stride;
        uniform float* uniform wdst_row2 = wdstp2 + y * stride;
        uniform float* uniform wdst_row3 = wdstp3 + y * stride;

        uniform const float* uniform t1_row = temp1 + y * stride;
        uniform const float* uniform t2_row = temp2 + y_mq * stride;
        uniform const unsigned int8* uniform src_bwd_row1 = srcp_bwd1 + y_pq * stride;
        uniform const unsigned int8* uniform src_bwd_row2 = srcp_bwd2 + y_pq * stride;
        uniform const unsigned int8* uniform src_bwd_row3 = srcp_bwd3 + y_pq * stride;
        uniform const unsigned int8* uniform src_fwd_row1 = srcp_fwd1 + y_mq * stride;
        uniform const unsigned int8* uniform src_fwd_row2 = srcp_fwd2 + y_mq * stride;
        uniform const unsigned int8* uniform src_fwd_row3 = srcp_fwd3 + y_mq * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform float u4 = t1_row[x];
            uniform float u4_mq = t2_row[CLAMPX(x - offset_x)];
            weight_row[x] += u4 + u4_mq;
            max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
            uniform int pq_x = CLAMPX(x + offset_x);
            uniform int mq_x = CLAMPX(x - offset_x);
            uniform float u1_pq_1 = src_bwd_row1[pq_x];
            uniform float u1_pq_2 = src_bwd_row2[pq_x];
            uniform float u1_pq_3 = src_bwd_row3[pq_x];
            uniform float u1_mq_1 = src_fwd_row1[mq_x];
            uniform float u1_mq_2 = src_fwd_row2[mq_x];
            uniform float u1_mq_3 = src_fwd_row3[mq_x];
            wdst_row1[x] += u4 * u1_pq_1 + u4_mq * u1_mq_1;
            wdst_row2[x] += u4 * u1_pq_2 + u4_mq * u1_mq_2;
            wdst_row3[x] += u4 * u1_pq_3 + u4_mq * u1_mq_3;
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                float u4 = t1_row[x];
                float u4_mq = t2_row[x - offset_x];
                weight_row[x] += u4 + u4_mq;
                max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
                float u1_pq_1 = src_bwd_row1[x + offset_x];
                float u1_pq_2 = src_bwd_row2[x + offset_x];
                float u1_pq_3 = src_bwd_row3[x + offset_x];
                float u1_mq_1 = src_fwd_row1[x - offset_x];
                float u1_mq_2 = src_fwd_row2[x - offset_x];
                float u1_mq_3 = src_fwd_row3[x - offset_x];
                wdst_row1[x] += u4 * u1_pq_1 + u4_mq * u1_mq_1;
                wdst_row2[x] += u4 * u1_pq_2 + u4_mq * u1_mq_2;
                wdst_row3[x] += u4 * u1_pq_3 + u4_mq * u1_mq_3;
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform float u4 = t1_row[x];
            uniform float u4_mq = t2_row[CLAMPX(x - offset_x)];
            weight_row[x] += u4 + u4_mq;
            max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
            uniform int pq_x = CLAMPX(x + offset_x);
            uniform int mq_x = CLAMPX(x - offset_x);
            uniform float u1_pq_1 = src_bwd_row1[pq_x];
            uniform float u1_pq_2 = src_bwd_row2[pq_x];
            uniform float u1_pq_3 = src_bwd_row3[pq_x];
            uniform float u1_mq_1 = src_fwd_row1[mq_x];
            uniform float u1_mq_2 = src_fwd_row2[mq_x];
            uniform float u1_mq_3 = src_fwd_row3[mq_x];
            wdst_row1[x] += u4 * u1_pq_1 + u4_mq * u1_mq_1;
            wdst_row2[x] += u4 * u1_pq_2 + u4_mq * u1_mq_2;
            wdst_row3[x] += u4 * u1_pq_3 + u4_mq * u1_mq_3;
        }
    }
}

export void nlmAccumulationCh3_u16(
    uniform float weightp[],
    uniform float wdstp1[],
    uniform float wdstp2[],
    uniform float wdstp3[],
    uniform float max_weightp[],
    uniform const unsigned int16 srcp_bwd1[],
    uniform const unsigned int16 srcp_bwd2[],
    uniform const unsigned int16 srcp_bwd3[],
    uniform const unsigned int16 srcp_fwd1[],
    uniform const unsigned int16 srcp_fwd2[],
    uniform const unsigned int16 srcp_fwd3[],
    uniform const float temp1[],
    uniform const float temp2[],
    uniform int offset_x,
    uniform int offset_y,
    uniform int width,
    uniform int height,
    uniform int stride
) {
    uniform int start_x = min(abs(offset_x), width);
    uniform int end_x = max(start_x, width - abs(offset_x));

    for (uniform int y = 0; y < height; y++) {
        uniform int y_mq = CLAMPY(y - offset_y);
        uniform int y_pq = CLAMPY(y + offset_y);

        uniform float* uniform weight_row = weightp + y * stride;
        uniform float* uniform max_weight_row = max_weightp + y * stride;
        uniform float* uniform wdst_row1 = wdstp1 + y * stride;
        uniform float* uniform wdst_row2 = wdstp2 + y * stride;
        uniform float* uniform wdst_row3 = wdstp3 + y * stride;

        uniform const float* uniform t1_row = temp1 + y * stride;
        uniform const float* uniform t2_row = temp2 + y_mq * stride;
        uniform const unsigned int16* uniform src_bwd_row1 = srcp_bwd1 + y_pq * stride;
        uniform const unsigned int16* uniform src_bwd_row2 = srcp_bwd2 + y_pq * stride;
        uniform const unsigned int16* uniform src_bwd_row3 = srcp_bwd3 + y_pq * stride;
        uniform const unsigned int16* uniform src_fwd_row1 = srcp_fwd1 + y_mq * stride;
        uniform const unsigned int16* uniform src_fwd_row2 = srcp_fwd2 + y_mq * stride;
        uniform const unsigned int16* uniform src_fwd_row3 = srcp_fwd3 + y_mq * stride;

        for (uniform int x = 0; x < start_x; x++) {
            uniform float u4 = t1_row[x];
            uniform float u4_mq = t2_row[CLAMPX(x - offset_x)];
            weight_row[x] += u4 + u4_mq;
            max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
            uniform int pq_x = CLAMPX(x + offset_x);
            uniform int mq_x = CLAMPX(x - offset_x);
            uniform float u1_pq_1 = src_bwd_row1[pq_x];
            uniform float u1_pq_2 = src_bwd_row2[pq_x];
            uniform float u1_pq_3 = src_bwd_row3[pq_x];
            uniform float u1_mq_1 = src_fwd_row1[mq_x];
            uniform float u1_mq_2 = src_fwd_row2[mq_x];
            uniform float u1_mq_3 = src_fwd_row3[mq_x];
            wdst_row1[x] += u4 * u1_pq_1 + u4_mq * u1_mq_1;
            wdst_row2[x] += u4 * u1_pq_2 + u4_mq * u1_mq_2;
            wdst_row3[x] += u4 * u1_pq_3 + u4_mq * u1_mq_3;
        }

        if (end_x > start_x) {
            foreach (x = start_x... end_x) {
                float u4 = t1_row[x];
                float u4_mq = t2_row[x - offset_x];
                weight_row[x] += u4 + u4_mq;
                max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
                float u1_pq_1 = src_bwd_row1[x + offset_x];
                float u1_pq_2 = src_bwd_row2[x + offset_x];
                float u1_pq_3 = src_bwd_row3[x + offset_x];
                float u1_mq_1 = src_fwd_row1[x - offset_x];
                float u1_mq_2 = src_fwd_row2[x - offset_x];
                float u1_mq_3 = src_fwd_row3[x - offset_x];
                wdst_row1[x] += u4 * u1_pq_1 + u4_mq * u1_mq_1;
                wdst_row2[x] += u4 * u1_pq_2 + u4_mq * u1_mq_2;
                wdst_row3[x] += u4 * u1_pq_3 + u4_mq * u1_mq_3;
            }
        }

        for (uniform int x = end_x; x < width; x++) {
            uniform float u4 = t1_row[x];
            uniform float u4_mq = t2_row[CLAMPX(x - offset_x)];
            weight_row[x] += u4 + u4_mq;
            max_weight_row[x] = max(max(u4, u4_mq), max_weight_row[x]);
            uniform int pq_x = CLAMPX(x + offset_x);
            uniform int mq_x = CLAMPX(x - offset_x);
            uniform float u1_pq_1 = src_bwd_row1[pq_x];
            uniform float u1_pq_2 = src_bwd_row2[pq_x];
            uniform float u1_pq_3 = src_bwd_row3[pq_x];
            uniform float u1_mq_1 = src_fwd_row1[mq_x];
            uniform float u1_mq_2 = src_fwd_row2[mq_x];
            uniform float u1_mq_3 = src_fwd_row3[mq_x];
            wdst_row1[x] += u4 * u1_pq_1 + u4_mq * u1_mq_1;
            wdst_row2[x] += u4 * u1_pq_2 + u4_mq * u1_mq_2;
            wdst_row3[x] += u4 * u1_pq_3 + u4_mq * u1_mq_3;
        }
    }
}

/* =========================================================================
   Finish Kernels (1 Channel)
   ========================================================================= */

export void nlmFinishCh1_f32(
    uniform float dstp[],
    uniform const float srcp[],
    uniform const float weightp[],
    uniform const float wdstp[],
    uniform const float max_weightp[],
    uniform float wref,
    uniform int width,
    uniform int height,
    uniform int stride
) {
    for (uniform int y = 0; y < height; y++) {
        uniform float* uniform dst_row = dstp + y * stride;
        uniform const float* uniform src_row = srcp + y * stride;
        uniform const float* uniform weight_row = weightp + y * stride;
        uniform const float* uniform wdst_row = wdstp + y * stride;
        uniform const float* uniform max_weight_row = max_weightp + y * stride;

        foreach (x = 0 ... width) {
            float multiplier = wref * max_weight_row[x];
            float denom = multiplier + weight_row[x];
            dst_row[x] = (denom > 0.0f) ? (multiplier * src_row[x] + wdst_row[x]) / denom : src_row[x];
        }
    }
}

export void nlmFinishCh1_u8(
    uniform unsigned int8 dstp[],
    uniform const unsigned int8 srcp[],
    uniform const float weightp[],
    uniform const float wdstp[],
    uniform const float max_weightp[],
    uniform float wref,
    uniform int width,
    uniform int height,
    uniform int stride,
    uniform int peak
) {
    for (uniform int y = 0; y < height; y++) {
        uniform unsigned int8* uniform dst_row = dstp + y * stride;
        uniform const unsigned int8* uniform src_row = srcp + y * stride;
        uniform const float* uniform weight_row = weightp + y * stride;
        uniform const float* uniform wdst_row = wdstp + y * stride;
        uniform const float* uniform max_weight_row = max_weightp + y * stride;

        foreach (x = 0 ... width) {
            float multiplier = wref * max_weight_row[x];
            float denom = multiplier + weight_row[x];
            float val = (denom > 0.0f) ? (multiplier * src_row[x] + wdst_row[x]) / denom : (float)src_row[x];
            dst_row[x] = clamp((int)round(val), 0, peak);
        }
    }
}

export void nlmFinishCh1_u16(
    uniform unsigned int16 dstp[],
    uniform const unsigned int16 srcp[],
    uniform const float weightp[],
    uniform const float wdstp[],
    uniform const float max_weightp[],
    uniform float wref,
    uniform int width,
    uniform int height,
    uniform int stride,
    uniform int peak
) {
    for (uniform int y = 0; y < height; y++) {
        uniform unsigned int16* uniform dst_row = dstp + y * stride;
        uniform const unsigned int16* uniform src_row = srcp + y * stride;
        uniform const float* uniform weight_row = weightp + y * stride;
        uniform const float* uniform wdst_row = wdstp + y * stride;
        uniform const float* uniform max_weight_row = max_weightp + y * stride;

        foreach (x = 0 ... width) {
            float multiplier = wref * max_weight_row[x];
            float denom = multiplier + weight_row[x];
            float val = (denom > 0.0f) ? (multiplier * src_row[x] + wdst_row[x]) / denom : (float)src_row[x];
            dst_row[x] = clamp((int)round(val), 0, peak);
        }
    }
}

/* =========================================================================
   Finish Kernels (2 Channels)
   ========================================================================= */

export void nlmFinishCh2_f32(
    uniform float dstp1[],
    uniform float dstp2[],
    uniform const float srcp1[],
    uniform const float srcp2[],
    uniform const float weightp[],
    uniform const float wdstp1[],
    uniform const float wdstp2[],
    uniform const float max_weightp[],
    uniform float wref,
    uniform int width,
    uniform int height,
    uniform int stride
) {
    for (uniform int y = 0; y < height; y++) {
        uniform float* uniform dst_row1 = dstp1 + y * stride;
        uniform float* uniform dst_row2 = dstp2 + y * stride;
        uniform const float* uniform src_row1 = srcp1 + y * stride;
        uniform const float* uniform src_row2 = srcp2 + y * stride;
        uniform const float* uniform weight_row = weightp + y * stride;
        uniform const float* uniform wdst_row1 = wdstp1 + y * stride;
        uniform const float* uniform wdst_row2 = wdstp2 + y * stride;
        uniform const float* uniform max_weight_row = max_weightp + y * stride;

        foreach (x = 0 ... width) {
            float multiplier = wref * max_weight_row[x];
            float denom = multiplier + weight_row[x];
            if (denom > 0.0f) {
                float inv_denom = 1.0f / denom;
                dst_row1[x] = (multiplier * src_row1[x] + wdst_row1[x]) * inv_denom;
                dst_row2[x] = (multiplier * src_row2[x] + wdst_row2[x]) * inv_denom;
            } else {
                dst_row1[x] = src_row1[x];
                dst_row2[x] = src_row2[x];
            }
        }
    }
}

export void nlmFinishCh2_u8(
    uniform unsigned int8 dstp1[],
    uniform unsigned int8 dstp2[],
    uniform const unsigned int8 srcp1[],
    uniform const unsigned int8 srcp2[],
    uniform const float weightp[],
    uniform const float wdstp1[],
    uniform const float wdstp2[],
    uniform const float max_weightp[],
    uniform float wref,
    uniform int width,
    uniform int height,
    uniform int stride,
    uniform int peak
) {
    for (uniform int y = 0; y < height; y++) {
        uniform unsigned int8* uniform dst_row1 = dstp1 + y * stride;
        uniform unsigned int8* uniform dst_row2 = dstp2 + y * stride;
        uniform const unsigned int8* uniform src_row1 = srcp1 + y * stride;
        uniform const unsigned int8* uniform src_row2 = srcp2 + y * stride;
        uniform const float* uniform weight_row = weightp + y * stride;
        uniform const float* uniform wdst_row1 = wdstp1 + y * stride;
        uniform const float* uniform wdst_row2 = wdstp2 + y * stride;
        uniform const float* uniform max_weight_row = max_weightp + y * stride;

        foreach (x = 0 ... width) {
            float multiplier = wref * max_weight_row[x];
            float denom = multiplier + weight_row[x];
            if (denom > 0.0f) {
                float inv_denom = 1.0f / denom;
                float val1 = (multiplier * src_row1[x] + wdst_row1[x]) * inv_denom;
                float val2 = (multiplier * src_row2[x] + wdst_row2[x]) * inv_denom;
                dst_row1[x] = clamp((int)round(val1), 0, peak);
                dst_row2[x] = clamp((int)round(val2), 0, peak);
            } else {
                dst_row1[x] = src_row1[x];
                dst_row2[x] = src_row2[x];
            }
        }
    }
}

export void nlmFinishCh2_u16(
    uniform unsigned int16 dstp1[],
    uniform unsigned int16 dstp2[],
    uniform const unsigned int16 srcp1[],
    uniform const unsigned int16 srcp2[],
    uniform const float weightp[],
    uniform const float wdstp1[],
    uniform const float wdstp2[],
    uniform const float max_weightp[],
    uniform float wref,
    uniform int width,
    uniform int height,
    uniform int stride,
    uniform int peak
) {
    for (uniform int y = 0; y < height; y++) {
        uniform unsigned int16* uniform dst_row1 = dstp1 + y * stride;
        uniform unsigned int16* uniform dst_row2 = dstp2 + y * stride;
        uniform const unsigned int16* uniform src_row1 = srcp1 + y * stride;
        uniform const unsigned int16* uniform src_row2 = srcp2 + y * stride;
        uniform const float* uniform weight_row = weightp + y * stride;
        uniform const float* uniform wdst_row1 = wdstp1 + y * stride;
        uniform const float* uniform wdst_row2 = wdstp2 + y * stride;
        uniform const float* uniform max_weight_row = max_weightp + y * stride;

        foreach (x = 0 ... width) {
            float multiplier = wref * max_weight_row[x];
            float denom = multiplier + weight_row[x];
            if (denom > 0.0f) {
                float inv_denom = 1.0f / denom;
                float val1 = (multiplier * src_row1[x] + wdst_row1[x]) * inv_denom;
                float val2 = (multiplier * src_row2[x] + wdst_row2[x]) * inv_denom;
                dst_row1[x] = clamp((int)round(val1), 0, peak);
                dst_row2[x] = clamp((int)round(val2), 0, peak);
            } else {
                dst_row1[x] = src_row1[x];
                dst_row2[x] = src_row2[x];
            }
        }
    }
}

/* =========================================================================
   Finish Kernels (3 Channels)
   ========================================================================= */

export void nlmFinishCh3_f32(
    uniform float dstp1[],
    uniform float dstp2[],
    uniform float dstp3[],
    uniform const float srcp1[],
    uniform const float srcp2[],
    uniform const float srcp3[],
    uniform const float weightp[],
    uniform const float wdstp1[],
    uniform const float wdstp2[],
    uniform const float wdstp3[],
    uniform const float max_weightp[],
    uniform float wref,
    uniform int width,
    uniform int height,
    uniform int stride
) {
    for (uniform int y = 0; y < height; y++) {
        uniform float* uniform dst_row1 = dstp1 + y * stride;
        uniform float* uniform dst_row2 = dstp2 + y * stride;
        uniform float* uniform dst_row3 = dstp3 + y * stride;
        uniform const float* uniform src_row1 = srcp1 + y * stride;
        uniform const float* uniform src_row2 = srcp2 + y * stride;
        uniform const float* uniform src_row3 = srcp3 + y * stride;
        uniform const float* uniform weight_row = weightp + y * stride;
        uniform const float* uniform wdst_row1 = wdstp1 + y * stride;
        uniform const float* uniform wdst_row2 = wdstp2 + y * stride;
        uniform const float* uniform wdst_row3 = wdstp3 + y * stride;
        uniform const float* uniform max_weight_row = max_weightp + y * stride;

        foreach (x = 0 ... width) {
            float multiplier = wref * max_weight_row[x];
            float denom = multiplier + weight_row[x];
            if (denom > 0.0f) {
                float inv_denom = 1.0f / denom;
                dst_row1[x] = (multiplier * src_row1[x] + wdst_row1[x]) * inv_denom;
                dst_row2[x] = (multiplier * src_row2[x] + wdst_row2[x]) * inv_denom;
                dst_row3[x] = (multiplier * src_row3[x] + wdst_row3[x]) * inv_denom;
            } else {
                dst_row1[x] = src_row1[x];
                dst_row2[x] = src_row2[x];
                dst_row3[x] = src_row3[x];
            }
        }
    }
}

export void nlmFinishCh3_u8(
    uniform unsigned int8 dstp1[],
    uniform unsigned int8 dstp2[],
    uniform unsigned int8 dstp3[],
    uniform const unsigned int8 srcp1[],
    uniform const unsigned int8 srcp2[],
    uniform const unsigned int8 srcp3[],
    uniform const float weightp[],
    uniform const float wdstp1[],
    uniform const float wdstp2[],
    uniform const float wdstp3[],
    uniform const float max_weightp[],
    uniform float wref,
    uniform int width,
    uniform int height,
    uniform int stride,
    uniform int peak
) {
    for (uniform int y = 0; y < height; y++) {
        uniform unsigned int8* uniform dst_row1 = dstp1 + y * stride;
        uniform unsigned int8* uniform dst_row2 = dstp2 + y * stride;
        uniform unsigned int8* uniform dst_row3 = dstp3 + y * stride;
        uniform const unsigned int8* uniform src_row1 = srcp1 + y * stride;
        uniform const unsigned int8* uniform src_row2 = srcp2 + y * stride;
        uniform const unsigned int8* uniform src_row3 = srcp3 + y * stride;
        uniform const float* uniform weight_row = weightp + y * stride;
        uniform const float* uniform wdst_row1 = wdstp1 + y * stride;
        uniform const float* uniform wdst_row2 = wdstp2 + y * stride;
        uniform const float* uniform wdst_row3 = wdstp3 + y * stride;
        uniform const float* uniform max_weight_row = max_weightp + y * stride;

        foreach (x = 0 ... width) {
            float multiplier = wref * max_weight_row[x];
            float denom = multiplier + weight_row[x];
            if (denom > 0.0f) {
                float inv_denom = 1.0f / denom;
                float val1 = (multiplier * src_row1[x] + wdst_row1[x]) * inv_denom;
                float val2 = (multiplier * src_row2[x] + wdst_row2[x]) * inv_denom;
                float val3 = (multiplier * src_row3[x] + wdst_row3[x]) * inv_denom;
                dst_row1[x] = clamp((int)round(val1), 0, peak);
                dst_row2[x] = clamp((int)round(val2), 0, peak);
                dst_row3[x] = clamp((int)round(val3), 0, peak);
            } else {
                dst_row1[x] = src_row1[x];
                dst_row2[x] = src_row2[x];
                dst_row3[x] = src_row3[x];
            }
        }
    }
}

export void nlmFinishCh3_u16(
    uniform unsigned int16 dstp1[],
    uniform unsigned int16 dstp2[],
    uniform unsigned int16 dstp3[],
    uniform const unsigned int16 srcp1[],
    uniform const unsigned int16 srcp2[],
    uniform const unsigned int16 srcp3[],
    uniform const float weightp[],
    uniform const float wdstp1[],
    uniform const float wdstp2[],
    uniform const float wdstp3[],
    uniform const float max_weightp[],
    uniform float wref,
    uniform int width,
    uniform int height,
    uniform int stride,
    uniform int peak
) {
    for (uniform int y = 0; y < height; y++) {
        uniform unsigned int16* uniform dst_row1 = dstp1 + y * stride;
        uniform unsigned int16* uniform dst_row2 = dstp2 + y * stride;
        uniform unsigned int16* uniform dst_row3 = dstp3 + y * stride;
        uniform const unsigned int16* uniform src_row1 = srcp1 + y * stride;
        uniform const unsigned int16* uniform src_row2 = srcp2 + y * stride;
        uniform const unsigned int16* uniform src_row3 = srcp3 + y * stride;
        uniform const float* uniform weight_row = weightp + y * stride;
        uniform const float* uniform wdst_row1 = wdstp1 + y * stride;
        uniform const float* uniform wdst_row2 = wdstp2 + y * stride;
        uniform const float* uniform wdst_row3 = wdstp3 + y * stride;
        uniform const float* uniform max_weight_row = max_weightp + y * stride;

        foreach (x = 0 ... width) {
            float multiplier = wref * max_weight_row[x];
            float denom = multiplier + weight_row[x];
            if (denom > 0.0f) {
                float inv_denom = 1.0f / denom;
                float val1 = (multiplier * src_row1[x] + wdst_row1[x]) * inv_denom;
                float val2 = (multiplier * src_row2[x] + wdst_row2[x]) * inv_denom;
                float val3 = (multiplier * src_row3[x] + wdst_row3[x]) * inv_denom;
                dst_row1[x] = clamp((int)round(val1), 0, peak);
                dst_row2[x] = clamp((int)round(val2), 0, peak);
                dst_row3[x] = clamp((int)round(val3), 0, peak);
            } else {
                dst_row1[x] = src_row1[x];
                dst_row2[x] = src_row2[x];
                dst_row3[x] = src_row3[x];
            }
        }
    }
}
