#[cfg(test)]
mod tests {
    use prav_core::arena::Arena;
    use prav_core::decoder::DecodingState;
    use prav_core::topology::SquareGrid;

    #[test]
    fn test_batch_processing_logic() {
        let mut buffer = vec![0u8; 1024 * 1024];
        let mut arena = Arena::new(&mut buffer);
        let mut decoder = DecodingState::<SquareGrid, 8>::new(&mut arena, 8, 8, 1);

        // Setup 4 nodes in a single block (0, 1, 2, 3)
        // Make them all active
        unsafe {
            let block = decoder.blocks_state.get_unchecked_mut(0);
            block.boundary = 0b1111;
            block.occupied = 0b1111;
        }

        // Set defects at 0 and 3. They should be connected via 0-1-2-3 chain.
        decoder.defect_mask[0] = 0b1001; // Bits 0 and 3

        // Execute process_block
        unsafe {
            decoder.process_block(0);
        }

        // We verify that defects at 0 and 3 are connected.
        // Since they are at the Top Edge (Row 0), they connect to the Boundary Node.
        // So find(0) == find(3) == boundary_node.

        let boundary_node = (decoder.parents.len() - 1) as u32;
        let r0 = decoder.find(0);
        let r1 = decoder.find(1);
        let r2 = decoder.find(2);
        let r3 = decoder.find(3);

        assert_eq!(r0, boundary_node, "Node 0 should be connected to boundary");
        assert_eq!(r3, boundary_node, "Node 3 should be connected to boundary");
        assert_eq!(r0, r3, "Node 0 and 3 should be merged (via boundary)");
        assert_eq!(r0, r1, "Node 0 and 1 should be merged");
        assert_eq!(r0, r2, "Node 0 and 2 should be merged");
    }
}
