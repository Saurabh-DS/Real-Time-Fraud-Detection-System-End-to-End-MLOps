"""
Feast Entity Definitions
========================
Defines the primary entities (User, Merchant) used for feature mapping and lookups.
"""

from feast import Entity

# Primary entity: User
# All behavioral features are computed per user
user = Entity(
    name="user_id",
    description="Unique identifier for a customer/cardholder",
    join_keys=["user_id"],
)

# Secondary entity: Merchant
# Used for merchant reputation features
merchant = Entity(
    name="merchant_id", 
    description="Unique identifier for a merchant",
    join_keys=["merchant_id"],
)
