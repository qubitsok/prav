fn compute_stripe_ranges(total_blocks: usize, num_threads: usize) -> Vec<(usize, usize)> {
    let align = 8;
    let blocks_per_thread = (total_blocks + num_threads - 1) / num_threads;
    let aligned_per_thread = ((blocks_per_thread + align - 1) / align) * align;

    let mut ranges = Vec::new();
    let mut current = 0;
    for _ in 0..num_threads {
        let end = (current + aligned_per_thread).min(total_blocks);
        if current < end {
            ranges.push((current, end));
        } else {
            ranges.push((total_blocks, total_blocks));
        }
        current = end;
    }
    ranges
}

#[test]
fn test_stripe_alignment() {
    let stripes = compute_stripe_ranges(100, 4);
    assert_eq!(stripes, vec![(0, 32), (32, 64), (64, 96), (96, 100)]);

    for (start, _) in stripes {
        assert_eq!(start % 8, 0, "Start index {} not aligned to 8", start);
    }
}

#[test]
fn test_stripe_alignment_small() {
    let stripes = compute_stripe_ranges(10, 4);
    assert_eq!(stripes, vec![(0, 8), (8, 10), (10, 10), (10, 10)]);
    for (start, end) in stripes {
        if start != end {
            assert_eq!(start % 8, 0);
        }
    }
}
