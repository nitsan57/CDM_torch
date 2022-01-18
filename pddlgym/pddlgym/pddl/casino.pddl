(define (domain casino)
  (:requirements :strips :typing)
  (:types location prize1 prize2 prize3)
  (:predicates
     (At ?loc - location)
     (IsCasino ?loc - location)
     (MoveTo ?loc - location)
     (GetPrize1 ?p1 - prize1)
     (HavePrize1 ?p1 - prize1)
     (GetPrize2 ?p2 - prize2)
     (HavePrize2 ?p2 - prize2)
     (GetPrize3 ?p3 - prize3)
     (HavePrize3 ?p3 - prize3)
  )

  ; (:actions MoveTo GetPrize1 GetPrize2 GetPrize3)

  (:action MoveTo
    :parameters (?sloc - location ?eloc - location)
    :precondition (and (MoveTo ?eloc)
                       (At ?sloc)
                  )
    :effect (and (not (At ?sloc))
                 (At ?eloc)
            )
  )

  (:action GetPrize1
    :parameters (?prize - prize1 ?loc - location)
    :precondition (and (GetPrize1 ?prize)
                       (At ?loc)
                       (IsCasino ?loc)
                       (not (HavePrize1 ?prize))
                  )
    :effect (and (HavePrize1 ?prize)
            )
  )

  (:action GetPrize2
    :parameters (?prize - prize2 ?loc - location)
    :precondition (and (GetPrize2 ?prize)
                       (At ?loc)
                       (IsCasino ?loc)
                       (not (HavePrize2 ?prize))
                  )
    :effect (and (HavePrize2 ?prize)
            )
  )

  (:action GetPrize3
    :parameters (?prize - prize3 ?loc - location)
    :precondition (and (GetPrize3 ?prize)
                       (At ?loc)
                       (IsCasino ?loc)
                       (not (HavePrize3 ?prize))
                  )
    :effect (and (HavePrize3 ?prize)
            )
  )

)