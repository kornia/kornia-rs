#!/usr/bin/env python3
"""Create a scratch-only, source-exact pydegensac seven-point C harness."""
import argparse
from hashlib import sha256
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pydegensac-root", type=Path, required=True)
parser.add_argument("--out", type=Path, required=True)
args = parser.parse_args()
ROOT = args.pydegensac_root / "src/pydegensac/degensac"
OUT = args.out
OUT.mkdir(parents=True, exist_ok=True)


def extract(source: str, signature: str) -> str:
    start = source.index(signature)
    brace = source.index("{", start)
    depth = 0
    for end in range(brace, len(source)):
        if source[end] == "{":
            depth += 1
        elif source[end] == "}":
            depth -= 1
            if depth == 0:
                return source[start : end + 1]
    raise ValueError(signature)


ftools = (ROOT / "Ftools.c").read_text()
utools = (ROOT / "utools.c").read_text()
hashes = {
    "Ftools.c": sha256(ftools.encode()).hexdigest(),
    "utools.c": sha256(utools.encode()).hexdigest(),
}
functions = "\n\n".join(
    [
        extract(ftools, "void slcm("),
        extract(ftools, "int rroots3 ("),
        extract(utools, "int nullspace("),
    ]
)
prefix = r'''/* Generated from current pydegensac functions; scratch-only. */
#include <math.h>
#include <stddef.h>
#include <string.h>
#define pit 1.0471975511965967
#define a11 (*A)
#define a12 (*(A+1))
#define a13 (*(A+2))
#define a21 (*(A+3))
#define a22 (*(A+4))
#define a23 (*(A+5))
#define a31 (*(A+6))
#define a32 (*(A+7))
#define a33 (*(A+8))
#define b11 (*B)
#define b12 (*(B+1))
#define b13 (*(B+2))
#define b21 (*(B+3))
#define b22 (*(B+4))
#define b23 (*(B+5))
#define b31 (*(B+6))
#define b32 (*(B+7))
#define b33 (*(B+8))
#define rr_a (*po)
#define rr_d (*(po + 3))
'''
wrapper = r'''

static int pyd7_solve(const double *m, int normalize, double *out) {
    double x1[7][2], x2[7][2];
    double A[81] = {0.0}, basis[81], poly[4], roots[3];
    int workspace[18], i, j, n;
    double c1x = 0, c1y = 0, c2x = 0, c2y = 0, r1 = 0, r2 = 0, s1 = 1, s2 = 1;
    for (i = 0; i < 7; ++i) {
        x1[i][0] = m[4*i]; x1[i][1] = m[4*i+1];
        x2[i][0] = m[4*i+2]; x2[i][1] = m[4*i+3];
        c1x += x1[i][0]; c1y += x1[i][1]; c2x += x2[i][0]; c2y += x2[i][1];
    }
    if (normalize) {
        c1x /= 7; c1y /= 7; c2x /= 7; c2y /= 7;
        for (i = 0; i < 7; ++i) {
            r1 += hypot(x1[i][0]-c1x, x1[i][1]-c1y);
            r2 += hypot(x2[i][0]-c2x, x2[i][1]-c2y);
        }
        if (r1 <= 0 || r2 <= 0) return 0;
        s1 = sqrt(2.0) / (r1 / 7); s2 = sqrt(2.0) / (r2 / 7);
        for (i = 0; i < 7; ++i) {
            x1[i][0] = (x1[i][0]-c1x)*s1; x1[i][1] = (x1[i][1]-c1y)*s1;
            x2[i][0] = (x2[i][0]-c2x)*s2; x2[i][1] = (x2[i][1]-c2y)*s2;
        }
    }
    for (i = 0; i < 7; ++i) {
        double x=x1[i][0], y=x1[i][1], xp=x2[i][0], yp=x2[i][1];
        A[9*i]=xp*x; A[9*i+1]=xp*y; A[9*i+2]=xp;
        A[9*i+3]=yp*x; A[9*i+4]=yp*y; A[9*i+5]=yp;
        A[9*i+6]=x; A[9*i+7]=y; A[9*i+8]=1;
    }
    if (nullspace(A, basis, 9, workspace) != 2) return 0;
    slcm(basis, basis+9, poly);
    n = rroots3(poly, roots);
    for (i = 0; i < n; ++i) {
        double F[9], G[9], norm = 0;
        for (j = 0; j < 9; ++j) F[j] = basis[j]*roots[i] + basis[9+j]*(1-roots[i]);
        if (normalize) {
            /* T2^T F T1 for T=[[s,0,-scx],[0,s,-scy],[0,0,1]]. */
            G[0]=s2*s1*F[0]; G[1]=s2*s1*F[1];
            G[2]=s2*(F[2]-s1*c1x*F[0]-s1*c1y*F[1]);
            G[3]=s2*s1*F[3]; G[4]=s2*s1*F[4];
            G[5]=s2*(F[5]-s1*c1x*F[3]-s1*c1y*F[4]);
            G[6]=s1*(F[6]-s2*c2x*F[0]-s2*c2y*F[3]);
            G[7]=s1*(F[7]-s2*c2x*F[1]-s2*c2y*F[4]);
            G[8]=F[8]-s1*c1x*F[6]-s1*c1y*F[7]-s2*c2x*(F[2]-s1*c1x*F[0]-s1*c1y*F[1])-s2*c2y*(F[5]-s1*c1x*F[3]-s1*c1y*F[4]);
            memcpy(F, G, sizeof(F));
        }
        if (normalize) {
            for (j = 0; j < 9; ++j) norm += F[j]*F[j];
            norm = sqrt(norm);
            for (j = 0; j < 9; ++j) out[9*i+j] = F[j]/norm;
        } else {
            memcpy(out + 9*i, F, sizeof(F));
        }
    }
    return n;
}

int pyd7_bench(const double *matches, int repetitions, int normalize, double *checksum) {
    double out[27];
    int i, j, n = 0;
    double sum = 0;
    for (i = 0; i < repetitions; ++i) {
        /* Compiler barrier matches Rust black_box: every fit reloads input. */
        __asm__ __volatile__("" : : "r"(matches) : "memory");
        n = pyd7_solve(matches, normalize, out);
        for (j = 0; j < n; ++j) sum += out[9*j];
    }
    *checksum = sum;
    return n;
}

int pyd7_once(const double *matches, int normalize, double *out) { return pyd7_solve(matches, normalize, out); }
'''
(OUT / "pydegensac_minimal.c").write_text(prefix + "\n" + functions + wrapper)
(OUT / "metadata.json").write_text(__import__("json").dumps({"sources": hashes, "functions": ["slcm", "rroots3", "nullspace"]}, indent=2) + "\n")
