
(define (problem generatedblocks) (:domain blocks)
  (:objects
        b0 - block
	b1 - block
	b10 - block
	b11 - block
	b2 - block
	b3 - block
	b4 - block
	b5 - block
	b6 - block
	b7 - block
	b8 - block
	b9 - block
  )
  (:init 
	(clear b0)
	(clear b5)
	(clear b9)
	(handempty)
	(on b0 b1)
	(on b10 b11)
	(on b1 b2)
	(on b2 b3)
	(on b3 b4)
	(on b5 b6)
	(on b6 b7)
	(on b7 b8)
	(on b9 b10)
	(ontable b11)
	(ontable b4)
	(ontable b8)
  )
  (:goal (and
	(on b11 b4)
	(on b4 b5)
	(on b5 b3)
	(on b3 b1)
	(ontable b1)
	(on b7 b6)
	(on b6 b9)
	(on b9 b2)
	(on b2 b10)
	(ontable b10)
	(on b0 b8)
	(ontable b8)))
)
