
(define (problem generatedblocks) (:domain blocks)
  (:objects
        b0 - block
	b1 - block
	b2 - block
	b3 - block
	b4 - block
	b5 - block
  )
  (:init 
	(clear b0)
	(clear b1)
	(clear b3)
	(clear b4)
	(handempty )
	(on b1 b2)
	(on b4 b5)
	(ontable b0)
	(ontable b2)
	(ontable b3)
	(ontable b5)
  )
  (:goal (and
	(on b0 b1)
	(on b1 b4)
	(ontable b4)
	(on b3 b5)
	(on b5 b2)
	(ontable b2)))
)
