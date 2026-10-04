from app.tools.criminal_rag import _CHARGEABLE_PUNISHMENT_RE


def test_recovers_abetment_and_same_manner_penalties():
    abetment = (
        "109. Punishment of abetment if the act abetted is committed in consequence. "
        "Whoever abets any offence shall, if the act abetted is committed in consequence "
        "and where no express provision is made, be punished with the punishment provided "
        "for the offence."
    )
    forged_use = (
        "471. Using as genuine a forged document. Whoever fraudulently uses as genuine any "
        "document which he knows to be forged shall be punished in the same manner as if he "
        "had forged such document."
    )
    assert _CHARGEABLE_PUNISHMENT_RE.search(abetment)
    assert _CHARGEABLE_PUNISHMENT_RE.search(forged_use)


def test_still_excludes_definitions_and_general_principles():
    definition = (
        "2. Punishment of offences committed within India. Every person shall be liable "
        "to punishment under this Code and not otherwise for every act or omission."
    )
    sentencing_rule = (
        "40. Offence. the word offence denotes a thing made punishable by this Code."
    )
    assert not _CHARGEABLE_PUNISHMENT_RE.search(definition)
    assert not _CHARGEABLE_PUNISHMENT_RE.search(sentencing_rule)
