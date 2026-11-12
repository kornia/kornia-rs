# Oracle rules for correctness checks

How to define, test, and report correctness contracts before adding an external oracle or parity check.

An oracle check compares the output of an implementation to an expected result from a trusted reference, and helps determine whether it works as intended. However, before adding such a check, it is important to establish exactly what should match and which differences are acceptable. Without a clearly defined contract, a test can end up enforcing assumptions that were never part of the original requirements.

This document provides a common set of guidelines for defining and evaluating oracle-based correctness checks.

## What is an oracle?

An oracle is a source of expected behavior or output that we use to verify our implementation. It can be:

- another implementation, such as OpenCV
- a simple scalar implementation used to check an optimized kernel
- a recorded fixture produced by a known reference.

An oracle is useful only when its source and comparison are understood. Record the implementation and version, settings, inputs, and how expected results were generated. Do not generate the expected result by calling the function under test.

The **contract** defines what the test compares, what counts as a pass, and what happens when the reference is unavailable. For example, saying that two outputs should be "close enough" does not establish a useful contract. A more meaningful contract would be:

"Output dimensions and channel order must match exactly, and every finite f32 output must satisfy the specified absolute and relative error bounds."

Not every part of an output needs to follow the same comparison rule. Shape and labels may require exact matches, while floating-point values may allow a small numerical difference. Define these requirements separately rather than treating the entire output as one comparison.

## Types of Correctness Contracts

The comparison method should be based on what the operation actually promises. Do not weaken an exact contract merely to make a failing test pass. The types of contracts are given below.

### Byte-exact

A byte-exact contract requires every specified output byte to match the reference.

Use this when an operation promises exact pixel or buffer output. State clearly which bytes are included in the comparison: logical pixels or the full buffer, row padding or no padding, and whether encoded-file metadata matters. Also specify shape, layout, and channel order.

Compare the specified bytes directly. A useful failure report includes the first differing pixel or offset, expected and actual values, and total number of differences.

Example:

```text
For these RGB u8 inputs, dimensions, channel order, and every logical output
pixel must match the named OpenCV version and interpolation mode.
```

### Bit-exact

Bit-exact comparison is stricter than ordinary numerical equality. It requires every specified output value to have exactly the same bit representation as the reference.

This distinction matters particularly for floating-point values. For example, positive zero and negative zero are numerically equal, but their bit patterns differ. Similarly, NaN values do not compare equal using ==, and different implementations may produce NaNs with different payloads.

When using a bit-exact contract, specify:

- **The floating-point type being compared.**
- **Which output fields must match.**
- **Whether raw IEEE-754 bit patterns are compared.**
- **The architectures and instruction paths for which the guarantee applies.**
- **Compiler settings and hardware instructions can affect floating-point  results, so these details matter.**

In Rust, to_bits() provides an explicit way to compare floating-point bit patterns.

Example:

```text
For the supported aarch64 path and fixed inputs, every output f32 must have
the same IEEE-754 bit pattern as the recorded reference.
```

### Tolerance-based

A tolerance-based contract allows small numerical differences between the implementation and its reference.

This is useful when identical numerical results are not guaranteed, such as when different architectures, fused multiply-add instructions, or floating-point evaluation orders produce slightly different results.

However, the allowed difference must be justified by the operation itself. A tolerance should not be copied from another test simply because it makes the comparison pass. A good tolerance-based contract should define the following:

- **Values being compared**: Specify which output values are included.
- **Error calculation**: Define how numerical differences are measured.
- **Threshold**: State the acceptable absolute and relative error.
- **Pass/fail criteria**: Clarify whether every element must satisfy the bound or whether an aggregate metric is used.
- **Special values**: Define how NaN, infinity, and zero are handled.
- **Exact requirements**: Identify properties such as shape, ordering, and channel layout that must still match exactly.

Common formulas for actual value `x` and reference `r` are:

```text
absolute error: abs(x - r)
relative error: abs(x - r) / abs(r)             #(undefined when r = 0; define handling explicitly)
combined bound: abs(x - r) <= atol + rtol * abs(r)
```

`atol` is the absolute tolerance; `rtol` is the relative tolerance. Relative error taken alone is unstable near zero. Mean or RMS error alone can hide a large outlier, so include a maximum or per-element bound when outliers matter.

Special values also need explicit handling. For example, a contract might require NaN and infinity to appear in exactly the same locations in both outputs, while applying the numerical tolerance only to finite values. This prevents NaN values from accidentally passing because a comparison returned false or was skipped.

Example:

```text
Shapes must match. Every finite f32 must satisfy
abs(actual - expected) <= atol + rtol * abs(expected), with atol and rtol
documented for this operation. NaN and infinity locations must match exactly.
```

### Semantic

A semantic contract focuses on whether the output satisfies the expected behavior rather than whether individual output elements match one-for-one. Use this for results where meaning matters more than representation.

Examples:

- **Detected feature locations must be within a specified distance of the reference, and the number of detections must satisfy a defined equality or acceptable range.**
- **Output points must use the documented coordinate frame and units.**
- **Output indices must be within valid bounds, and results must follow the documented ordering.**
- **Invalid inputs return the documented error.**

Statements such as "the result looks right" or "both implementations behave similarly" are not sufficient. Define the expected conditions, including relevant boundary cases, so that the test can evaluate them consistently.

Discrete properties such as shape, count, labels, and valid indices should still be checked exactly whenever they are part of the operation's requirements.


### Report-only

A report-only check calculates and displays comparison results without using them to determine whether the test passes or fails.

This is useful during early development, when investigating differences between implementations or collecting baseline measurements before agreeing on an acceptable threshold. When using this approach:

- **Clearly label the result as report-only or non-gating.**
- **Identify the oracle and inputs used.**
- **Include relevant comparison metrics.**
- **Record the environment and configuration.**
- **Make it clear that the result does not establish correctness.**

A report-only comparison should never be described as a parity pass.

If the project later decides to make the comparison a required correctness check, define and review the contract first. Only then should the test be changed from report-only to gating. A successful report-only run does not prove that an implementation is correct.

## Choose the Contract Before Adding the Check

Before introducing an oracle or external parity check, take the time to understand what the operation is expected to guarantee.

The following questions help establish a meaningful correctness contract:

1. **What behavior does the operation promise?** Does it require exact parity, allow bounded numerical differences, or guarantee a particular semantic property?
2. **What differences are acceptable?** Consider floating-point rounding, hardware architecture, algorithm settings, output representation, and execution order.
3. **What must remain identical?** Identify properties such as shape, dtype, channel order, labels, ordering, border behavior, and error handling.
4. **Which oracle should be used?** Record its name, version, configuration, and why it is a suitable reference.
5. **Which inputs should be tested?** Include ordinary cases as well as relevant boundary conditions, degenerate inputs, supported data types, and backend configurations.
6. **Can failures be understood?** The test should provide enough information to identify where the outputs differ and how significant the difference is.
7. **Is it clear whether the comparison actually ran?** Missing fixtures, unavailable hardware, or optional dependencies should never make a skipped check look like a successful comparison.

Use the strongest contract that matches the requirements. Different parts of the same output can have different contracts:

```text
Output dimensions and channel order: exact.
u8 pixels: byte-exact against the named reference.
f32 metrics: tolerance-based with operation-specific bounds.
Detected locations: semantic distance and count requirements.
Exploration metrics: report-only; not a correctness gate.
```

## Add a check

Once the contract is clear, follow these steps when implementing the check:

1. **Look for existing implementations and tests:** Start by checking the operation's crate, existing tests, and documentation. There may already be a suitable fixture, comparison helper, or reference implementation that can be reused. Reuse existing utilities when they fit, but do not copy tolerances from unrelated operations without understanding why they were chosen.

2. **Define the contract:** Document exactly what is being compared, which parts of the output are included, the contract type, and the pass/fail conditions. Include relevant edge cases and the configurations for which the comparison is expected to work.

3. **Verify the reference implementation:** Make sure the oracle is independent of the function being tested. Record how its expected results were generated, including the version, settings, and source of the reference data.

4. **Choose representative inputs:** Test more than just typical inputs. Include cases that are likely to expose implementation differences, such as boundary pixels, odd dimensions, zero values, rounding boundaries, degenerate inputs, and non-finite values where supported.

5. **Implement the comparison:** Keep structural checks, such as shape and channel order, separate from numerical comparisons. Make failure messages useful. They should identify the input, the location of the mismatch, and the expected and actual values or relevant error metrics.

6. **Run the check with the oracle enabled:** Confirm that the comparison actually executes and does not simply get skipped. For gating checks, verify that an intentional mismatch causes the test to fail. If a dependency, fixture, or hardware requirement is unavailable, make that status visible rather than reporting a successful run.

7. **Document any changes to the contract:** Changing the reference implementation, tolerance, or comparison method changes what the test considers correct. Explain why this change is necessary and include the evidence supporting it. These changes should be deliberate rather than adjustments made simply to get a passing test.

If the oracle depends on specific hardware or external reference data, document the exact command used to run the check and clearly state whether the comparison actually executed.

Follow the repository's test and verification requirements in [`CONTRIBUTING.md`](../CONTRIBUTING.md). If an oracle requires hardware or reference data, report the exact command and whether that comparison ran.

## Contract template

Fill this in before implementing a new external check:

```text
Operation / output:
Intended behavior:
Oracle (name, version, settings, source):
Oracle independence / validation:
Expected result generation method:
Inputs covered:
Comparison scope:
Contract type:
Pass/fail rule:
Special values and edge cases:
Supported backends / architectures / features:
Command to run:
How skips or unavailable references are reported:
Failure diagnostics:
Known limitations:
```

## Common failures

- **Oracle data was not loaded:** A skipped comparison is not a passing comparison. Always make execution status visible.
- **Tolerance was chosen to make a test pass:** Investigate the underlying numerical difference first. Only change the tolerance when the allowed error can be justified by the operation.
- **Average error hides an outlier:** Mean and RMS error can make an output look accurate even when individual values are far outside the expected range. Include a bound check when those outliers matter.
- **Floating point `==` comparison is treated as bit-exact:** Numerical equality does not guarantee identical bit patterns. Use explicit bit comparisons when bit identity is required.
- **A visual inspection is treated as a test:** It can help debug but cannot replace a repeatable comparison.
- **Report-only results are described as successful parity checks:** Reporting metrics is not the same as enforcing a correctness contract. Make the distinction clear.
- **Discrete behavior is put under a numeric tolerance:** Properties such as counts, indices, labels, shapes, and error categories should have their own explicit comparison rules.
- **A check is added before the contract is agreed:** This can introduce arbitrary thresholds and produce misleading results that appear correct without actually verifying the intended behavior.


## Repository examples

The following documents provide practical examples of how correctness contracts and oracle-based comparisons are handled in the repository:

- [`SIMD.md`](../crates/kornia-imgproc/SIMD.md) starts with a scalar
  correctness reference and gives concrete SIMD comparison and tolerance
  examples.
- [`benchmark-sift.md`](benchmark-sift.md) documents a bitwise contract,
  oracle inputs, optional reference setup, and mismatch debugging.
- [`find-contours-cv2-parity.md`](superpowers/specs/2026-05-06-find-contours-cv2-parity.md)
  records parity goals, fixtures, intermediate results, and root-cause work.

These examples are intended to show how the guidelines can be applied to individual operations. The exact contract will depend on what each operation promises and how its output is expected to behave.