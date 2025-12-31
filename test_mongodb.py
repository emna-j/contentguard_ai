
import asyncio
from pymongo import MongoClient
from datetime import datetime


async def test_async():
    from motor.motor_asyncio import AsyncIOMotorClient

    print(" Test connexion Motor (async)...")
    try:
        client = AsyncIOMotorClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client.contentguard
        coll = db.contents

        test_doc = {
            "filename": "test_image.jpg",
            "is_fake": False,
            "confidence": 85.5,
            "status": "approved",
            "timestamp": datetime.now().isoformat()
        }

        result = await coll.insert_one(test_doc)
        print(f"✓ Document inséré (async): {result.inserted_id}")

        # Read test
        count = await coll.count_documents({})
        print(f"✓ Total documents: {count}")

        # Delete test document
        await coll.delete_one({"_id": result.inserted_id})
        print(f"✓ Document test supprimé")

    except Exception as e:
        print(f" Erreur Motor: {e}")


def test_sync():
    """Test avec PyMongo (sync)"""
    print("\n Test connexion PyMongo (sync)...")
    try:
        client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=5000)
        db = client.contentguard
        coll = db.contents

        test_doc = {
            "filename": "test_sync.jpg",
            "is_fake": True,
            "confidence": 92.3,
            "status": "rejected",
            "timestamp": datetime.now().isoformat()
        }

        result = coll.insert_one(test_doc)
        print(f" Document inséré (sync): {result.inserted_id}")

        # Read test
        count = coll.count_documents({})
        print(f" Total documents: {count}")

        # List recent
        docs = list(coll.find().sort("_id", -1).limit(5))
        print(f" {len(docs)} derniers documents récupérés")

        for doc in docs:
            print(f"  - {doc['filename']}: {doc['status']} ({doc['confidence']}%)")

        # Delete test document
        coll.delete_one({"_id": result.inserted_id})
        print(f"✓ Document test supprimé")

    except Exception as e:
        print(f" Erreur PyMongo: {e}")
        print("  Assurez-vous que MongoDB est lancé: mongod --dbpath=./data")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST MONGODB - ContentGuard AI")
    print("=" * 60)

    # Test sync
    test_sync()

    # Test async
    print()
    asyncio.run(test_async())

    print("\n✓ Tests terminés !")