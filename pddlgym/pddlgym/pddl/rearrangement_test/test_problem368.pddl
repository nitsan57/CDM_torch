(define (problem rearrangement) 
    (:domain rearrangement)

    (:objects
    
	pawn-0 - moveable
	monkey-1 - moveable
	bear-2 - moveable
	robot - moveable
	loc-0-0 - static
	loc-0-1 - static
	loc-0-2 - static
	loc-1-0 - static
	loc-1-1 - static
	loc-1-2 - static
	loc-2-0 - static
	loc-2-1 - static
	loc-2-2 - static
	loc-3-0 - static
	loc-3-1 - static
	loc-3-2 - static
    )

    (:init
    
	(ispawn pawn-0)
	(ismonkey monkey-1)
	(isbear bear-2)
	(isrobot robot)
	(at pawn-0 loc-3-0)
	(at monkey-1 loc-1-2)
	(at bear-2 loc-0-2)
	(at robot loc-0-1)
	(handsfree robot)

    ; action literals
    
	(pick pawn-0)
	(place pawn-0)
	(pick monkey-1)
	(place monkey-1)
	(pick bear-2)
	(place bear-2)
	(moveto loc-0-0)
	(moveto loc-0-1)
	(moveto loc-0-2)
	(moveto loc-1-0)
	(moveto loc-1-1)
	(moveto loc-1-2)
	(moveto loc-2-0)
	(moveto loc-2-1)
	(moveto loc-2-2)
	(moveto loc-3-0)
	(moveto loc-3-1)
	(moveto loc-3-2)
    )

    (:goal (and  (at pawn-0 loc-3-2) ))
)
    