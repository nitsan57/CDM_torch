(define (domain baking)
    (:requirements :typing )
    (:types ingredient pan oven soap)
    (:predicates
        (isegg ?egg - ingredient)
        (isflour ?flour - ingredient)
        (panhasegg ?pan - pan)
        (panhasflour ?pan - pan)
        (panisclean ?pan - pan)
        (paninoven ?pan - pan)
        (inpan ?x - ingredient ?pan - pan)
        (inoven ?pan - pan ?oven - oven)
        (ovenisfull ?oven - oven)
        (hypothetical ?new - ingredient)
        (ismixed ?pan - pan)
        (iscake ?new - ingredient)
        (issouffle ?new - ingredient)
        (soapconsumed ?soap - soap)
        (putegginpan ?egg - ingredient ?pan - pan)
        (putflourinpan ?flour - ingredient ?pan - pan)
        (mix ?pan - pan)
        (putpaninoven ?pan - pan ?oven - oven)
        (removepanfromoven ?pan - pan)
        (bakecake ?new - ingredient ?oven - oven)
        (bakesouffle ?new - ingredient ?oven - oven)
        (cleanpan ?pan - pan ?soap - soap)
    )

    ; (:actions putegginpan putflourinpan mix putpaninoven removepanfromoven bakecake bakesouffle cleanpan)

    (:action putegginpan
    :parameters (?egg - ingredient ?pan - pan)
    :precondition (and (putegginpan ?egg ?pan)
                       (isegg ?egg)
                       (not (panhasegg ?pan))
                       (not (ismixed ?pan))
                       (panisclean ?pan)
                       (not (paninoven ?pan))
                  )
    :effect (and (panhasegg ?pan)
                 (inpan ?egg ?pan)
            )
    )

    (:action putflourinpan
    :parameters (?flour - ingredient ?pan - pan)
    :precondition (and (putflourinpan ?flour ?pan)
                       (isflour ?flour)
                       (not (panhasflour ?pan))
                       (not (ismixed ?pan))
                       (panisclean ?pan)
                       (not (paninoven ?pan))
                  )
    :effect (and (panhasflour ?pan)
                 (inpan ?flour ?pan)
            )
    )

    (:action mix
    :parameters (?egg - ingredient ?flour - ingredient ?pan - pan)
    :precondition (and (mix ?pan)
                       (inpan ?egg ?pan)
                       (inpan ?flour ?pan)
                       (isegg ?egg)
                       (isflour ?flour)
                       (not (paninoven ?pan))
                  )
    :effect (and (ismixed ?pan)
                 (not (isegg ?egg))
                 (not (isflour ?flour))
                 (not (inpan ?egg ?pan))
                 (not (inpan ?flour ?pan))
                 (not (panhasegg ?pan))
                 (not (panhasflour ?pan))
            )
    )

    (:action putpaninoven
    :parameters (?pan - pan ?oven - oven)
    :precondition (and (putpaninoven ?pan ?oven)
                       (not (ovenisfull ?oven))
                       (not (paninoven ?pan))
                  )
    :effect (and (ovenisfull ?oven)
                 (inoven ?pan ?oven)
                 (paninoven ?pan)
            )
    )

    (:action removepanfromoven
    :parameters (?pan - pan ?oven - oven)
    :precondition (and (removepanfromoven ?pan)
                       (inoven ?pan ?oven)
                  )
    :effect (and (not (ovenisfull ?oven))
                 (not (inoven ?pan ?oven))
                 (not (paninoven ?pan))
            )
    )

    (:action bakecake
    :parameters (?oven - oven ?pan - pan ?new - ingredient)
    :precondition (and (bakecake ?new ?oven)
                       (ismixed ?pan)
                       (inoven ?pan ?oven)
                       (hypothetical ?new)
                  )
    :effect (and (not (ismixed ?pan))
                 (not (panisclean ?pan))
                 (not (hypothetical ?new))
                 (iscake ?new)
            )
    )

    (:action bakesouffle
    :parameters (?oven - oven ?egg - ingredient ?pan - pan ?new - ingredient)
    :precondition (and (bakesouffle ?new ?oven)
                       (inpan ?egg ?pan)
                       (isegg ?egg)
                       (not (panhasflour ?pan))
                       (inoven ?pan ?oven)
                       (hypothetical ?new)
                  )
    :effect (and (not (isegg ?egg))
                 (not (inpan ?egg ?pan))
                 (not (panhasegg ?pan))
                 (not (panisclean ?pan))
                 (not (hypothetical ?new))
                 (issouffle ?new)
            )
    )

    (:action cleanpan
    :parameters (?pan - pan ?soap - soap)
    :precondition (and (cleanpan ?pan ?soap)
                       (not (soapconsumed ?soap))
                       (not (paninoven ?pan))
                  )
    :effect (and (panisclean ?pan)
                 (soapconsumed ?soap)
            )
    )

)
        