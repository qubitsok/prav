MEMORY
{
  /* VersatilePB loads at 0x10000 usually */
  FLASH : ORIGIN = 0x00010000, LENGTH = 1M
  RAM : ORIGIN = 0x00200000, LENGTH = 64M
}

_stack_top = ORIGIN(RAM) + LENGTH(RAM);