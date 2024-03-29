
(define (problem generatedblocks) (:domain blocks)
  (:objects
        b0 - block
	b1 - block
	b2 - block
	b3 - block
	b4 - block
	b5 - block
	b6 - block
	b7 - block
	b8 - block
  )
  (:init 
	(clear b0)
	(clear b2)
	(clear b4)
	(clear b6)
	(clear b8)
	(handempty )
	(on b0 b1)
	(on b2 b3)
	(on b4 b5)
	(on b6 b7)
	(ontable b1)
	(ontable b3)
	(ontable b5)
	(ontable b7)
	(ontable b8)
  )
  (:goal (and
	(on b8 b4)
	(ontable b4)
	(on b5 b2)
	(ontable b2)
	(on b7 b6)
	(on b6 b0)
	(ontable b0)
	(on b3 b1)
	(ontable b1)))
)
