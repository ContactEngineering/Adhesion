
from Adhesion.ReferenceSolutions import JKR


def test_penetration_radius_inverse():
    params = dict(radius=1., contact_modulus=1., work_of_adhesion=1.)
    assert abs(JKR.contact_radius(
        penetration=JKR.penetration(contact_radius=1., **params), **params) - 1
        ) < 1e-10