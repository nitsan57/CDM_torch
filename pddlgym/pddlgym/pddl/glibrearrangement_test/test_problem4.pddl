(define (problem glibrearrangement) 
    (:domain glibrearrangement)

    (:objects
    
	pawn-0 - moveable
	bear-1 - moveable
	pawn-2 - moveable
	bear-3 - moveable
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
	loc-4-0 - static
	loc-4-1 - static
	loc-4-2 - static
    )

    (:init
    
	(IsPawn pawn-0)
	(IsBear bear-1)
	(IsPawn pawn-2)
	(IsBear bear-3)
	(IsRobot robot)
	(At pawn-0 loc-3-0)
	(At bear-1 loc-3-0)
	(At pawn-2 loc-2-2)
	(At bear-3 loc-4-1)
	(At robot loc-3-0)
	(Handsfree robot)

    ; Action literals
    
	(Pick pawn-0)
	(Place pawn-0)
	(Pick bear-1)
	(Place bear-1)
	(Pick pawn-2)
	(Place pawn-2)
	(Pick bear-3)
	(Place bear-3)
	(MoveTo loc-0-0)
	(MoveTo loc-0-1)
	(MoveTo loc-0-2)
	(MoveTo loc-1-0)
	(MoveTo loc-1-1)
	(MoveTo loc-1-2)
	(MoveTo loc-2-0)
	(MoveTo loc-2-1)
	(MoveTo loc-2-2)
	(MoveTo loc-3-0)
	(MoveTo loc-3-1)
	(MoveTo loc-3-2)
	(MoveTo loc-4-0)
	(MoveTo loc-4-1)
	(MoveTo loc-4-2)
    )

    (:goal (and  (At pawn-0 loc-0-0)  (At bear-1 loc-3-1)  (At pawn-2 loc-4-0) ))
)
    