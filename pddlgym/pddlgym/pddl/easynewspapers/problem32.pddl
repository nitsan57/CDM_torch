
(define (problem easynewspaper) (:domain easynewspapers)
  (:objects
        loc-0 - loc
	loc-1 - loc
	loc-2 - loc
	loc-3 - loc
	paper-0 - paper
	paper-1 - paper
	paper-10 - paper
	paper-11 - paper
	paper-12 - paper
	paper-13 - paper
	paper-2 - paper
	paper-3 - paper
	paper-4 - paper
	paper-5 - paper
	paper-6 - paper
	paper-7 - paper
	paper-8 - paper
	paper-9 - paper
  )
  (:init 
	(at loc-0)
	(ishomebase loc-0)
	(unpacked paper-0)
	(unpacked paper-10)
	(unpacked paper-11)
	(unpacked paper-12)
	(unpacked paper-13)
	(unpacked paper-1)
	(unpacked paper-2)
	(unpacked paper-3)
	(unpacked paper-4)
	(unpacked paper-5)
	(unpacked paper-6)
	(unpacked paper-7)
	(unpacked paper-8)
	(unpacked paper-9)
	(wantspaper loc-1)
	(wantspaper loc-2)
	(wantspaper loc-3)
  )
  (:goal (and
	(satisfied loc-1)
	(satisfied loc-2)
	(satisfied loc-3)))
)
