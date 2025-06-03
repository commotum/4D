ARC-AGI (Collection @ [0,0,0,0])
└── Training (Collection @ [0,0,0,0])
    ├── Squares (Collection @ [0,0,0,0])
    ├── Circles (Collection)
    │   └── Circle 1 (Collection)
    │       ├── Circle-1.ts (.ts)
    │       │   ├── Setup
    │       │   │   ├── import statement line 1 (.ts string?)  @ [0,0,1,0] (import type { P5Task } from '../types')
    │       │   │   │   ├── import @ [0,1,1,0]
    │       │   │   │   ├── type @ [0,2,1,0]
    │       │   │   │   ├── {  } @ [0,3,1,0]
    │       │   │   │   │   └── P5Task @ [0,3,1,1]
    │       │   │   │   ├── from @ [0,4,1,0]
    │       │   │   │   └── '' @ [0,5,1,0]
    │       │   │   │       └── ../types @ [0,5,1,1]
    │       │   │   ├── import statement line 2 @ [0,0,2,0]
    │       │   │   │   ├── 
    │       │   ├──


├── Line 1                Code                      [0,0,1,0]
│   ├── "import"          Keyword                   [0,1,1,0]
│   ├── "type"            Keyword                   [0,2,1,0]
│   ├── "{ _ }"           Punctuation               [0,3,1,0]
│   │   └── "P5Task"      Identifier                [0,1,1,1]
│   ├── "from"            Keyword                   [0,4,1,0]
│   ├── "'../types'"      StringLiteral             [0,5,1,0]
│   ├── "⏎"               LineTerminator            [0,6,1,0]
├── Line 2                Code                      [0,0,2,0]
│   ├── "import"          Keyword                   [0,1,2,0]
│   ├── "p5"              Identifier                [0,2,2,0]
│   ├── "from"            Keyword                   [0,3,2,0]
│   ├── "'p5'"            StringLiteral             [0,4,2,0]
│   ├── "⏎"               LineTerminator            [0,5,2,0]
├── Line 3                Code                      [0,0,3,0]
│   ├── "⏎"               LineTerminator            [0,1,3,0]
├── Line 4                Code                      [0,0,4,0]
│   ├── "const"           Keyword                   [0,1,4,0]
│   ├── "task"            Identifier                [0,2,4,0]
│   ├── ":"               Punctuation               [0,3,4,0]
│   ├── "P5Task"          Identifier                [0,4,4,0]
│   ├── "="               Operator                  [0,5,4,0]
│   ├── "{ _ }"           Punctuation               [0,6,4,0]
│   │   ├── "width"       Identifier                [0,1,5,1]
│   │   ├── ":"           Punctuation               [0,2,5,1]
│   │   ├── "512"         NumericLiteral            [0,3,5,1]
│   │   ├── ","           Punctuation               [0,4,5,1]
│   │   ├── "height"      Identifier                [0,5,6,1]
│   │   ├── ":"           Punctuation               [0,6,6,1]
│   │   ├── "512"         NumericLiteral            [0,7,6,1]
│   │   ├── ","           Punctuation               [0,8,6,1]
│   │   ├── "draw"        Identifier                [0,9,7,1]
│   │   ├── ":"           Punctuation               [0,10,7,1]
│   │   ├── "(p: p5)"     Parameter                 [0,11,7,1]
│   │   ├── "=>"          Operator                  [0,12,7,1]
│   │   ├── "{ _ }"       Punctuation               [0,13,7,1]
│   │   │   ├── "// Set white background" Comment   [0,1,8,2]
│   │   │   ├── "⏎"       LineTerminator            [0,2,8,2]
│   │   │   ├── "p"       Identifier                [0,3,9,2]
│   │   │   ├── "."       Punctuation               [0,4,9,2]
│   │   │   ├── "background" Identifier             [0,5,9,2]
│   │   │   ├── "("       Punctuation               [0,6,9,2]
│   │   │   ├── "255"     NumericLiteral            [0,7,9,2]
│   │   │   ├── ")"       Punctuation               [0,8,9,2]
│   │   │   ├── "⏎"       LineTerminator            [0,9,9,2]
│   │   │   ├── "⏎"       LineTerminator            [0,10,10,2]
│   │   │   ├── "// Draw a black circle" Comment    [0,11,11,2]
│   │   │   ├── "⏎"       LineTerminator            [0,12,11,2]
│   │   │   ├── "// centered at (x,y)" Comment      [0,13,12,2]
│   │   │   ├── "⏎"       LineTerminator            [0,14,12,2]
│   │   │   ├── "// with d diameter" Comment        [0,15,13,2]
│   │   │   ├── "⏎"       LineTerminator            [0,16,13,2]
│   │   │   ├── "// p.circle(x, y, d)" Comment      [0,17,14,2]
│   │   │   ├── "⏎"       LineTerminator            [0,18,14,2]
│   │   │   ├── "p"       Identifier                [0,19,15,2]
│   │   │   ├── "."       Punctuation               [0,20,15,2]
│   │   │   ├── "fill"    Identifier                [0,21,15,2]
│   │   │   ├── "("       Punctuation               [0,22,15,2]
│   │   │   ├── "0"       NumericLiteral            [0,23,15,2]
│   │   │   ├── ")"       Punctuation               [0,24,15,2]
│   │   │   ├── "⏎"       LineTerminator            [0,25,15,2]
│   │   │   ├── "p"       Identifier                [0,26,16,2]
│   │   │   ├── "."       Punctuation               [0,27,16,2]
│   │   │   ├── "circle"  Identifier                [0,28,16,2]
│   │   │   ├── "("       Punctuation               [0,29,16,2]
│   │   │   ├── "-20"     NumericLiteral            [0,30,16,2]
│   │   │   ├── ","       Punctuation               [0,31,16,2]
│   │   │   ├── "-201"    NumericLiteral            [0,32,16,2]
│   │   │   ├── ","       Punctuation               [0,33,16,2]
│   │   │   ├── "88"      NumericLiteral            [0,34,16,2]
│   │   │   ├── ")"       Punctuation               [0,35,16,2]
│   │   │   ├── "⏎"       LineTerminator            [0,36,16,2]
│   │   ├── "⏎"           LineTerminator            [0,14,17,1]
│   ├── "⏎"               LineTerminator            [0,7,18,0]
├── Line 19               Code                      [0,0,19,0]
│   ├── "export"          Keyword                   [0,1,19,0]
│   ├── "default"         Keyword                   [0,2,19,0]
│   ├── "task"            Identifier                [0,3,19,0]
│   ├── "⏎"               LineTerminator            [0,4,19,0]
         




import type { P5Task } from '../types'
import p5 from 'p5'

const task: P5Task = {
  width: 512,
  height: 512,
  draw: (p: p5) => {
    // Set white background
    p.background(255)
    
    // Draw a black circle
    // centered at (x,y)
    // with d diameter
    // p.circle(x, y, d)
    p.fill(0)
    p.circle(-20, -201, 88)
  }
}

export default task


│   ├── att.py
│   ├── axes.md
│   ├── color.py
│   ├── D10.txt
│   ├── dual.md
│   ├── geometricLattice.py
│   ├── Lattice
│   │   ├── attention.md
│   │   ├── lattice.py
│   │   ├── notes.md
│   │   ├── progression.md
│   │   ├── Q.py
│   │   └── train.py
│   ├── lattice.py
│   ├── map.py
│   ├── MonSTER-Detailed.py
│   ├── MonSTER-V1.py
│   ├── rgb.py
│   ├── RoPE.py
│   ├── single.py
│   ├── test2.py
│   ├── test.py
│   ├── try.md
│   └── ww.py
├── __pycache__
│   ├── mink.cpython-310.pyc
│   └── MonSTER.cpython-310.pyc
├── README.md
├── Ref
│   ├── McCarthyOnTheUnconscious.md
│   ├── Quaternion Neural Network and Its Application.md
│   ├── Quaternion Recurrent Neural Networks.md
│   └── RoPE.md
└── test.py







import type { P5Task } from '../types'
import p5 from 'p5'

const task: P5Task = {
  width: 512,
  height: 512,
  draw: (p: p5) => {
    // Set white background
    p.background(255)
    
    // Draw a black circle
    // centered at (x,y)
    // with d diameter
    // p.circle(x, y, d)
    p.fill(0)
    p.circle(-20, -201, 88)
  }
}

export default task
