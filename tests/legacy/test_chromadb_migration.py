#!/usr/bin/env python3
"""
Test ChromaDB 1.0.15 migration and functionality.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

def test_chromadb_migration():
    """Test ChromaDB 1.0.15 functionality."""
    print("🔍 ChromaDB 1.0.15 Migration Test")
    print("=" * 50)
    
    try:
        import chromadb
        print(f"ChromaDB Version: {chromadb.__version__}")
        
        # Create temporary directory for testing
        temp_dir = tempfile.mkdtemp()
        print(f"Test directory: {temp_dir}")
        
        try:
            # Test 1: PersistentClient creation
            print("\n📁 Testing PersistentClient creation...")
            client = chromadb.PersistentClient(path=temp_dir)
            print("✅ PersistentClient created successfully")
            
            # Test 2: Collection creation
            print("\n📚 Testing collection creation...")
            collection = client.get_or_create_collection(
                name="test_memories",
                metadata={"description": "Test collection for Coda memories"}
            )
            print("✅ Collection created successfully")
            
            # Test 3: Adding documents
            print("\n📝 Testing document addition...")
            collection.add(
                ids=["mem1", "mem2", "mem3"],
                documents=[
                    "This is a test memory about Python programming",
                    "Another memory about machine learning",
                    "A third memory about voice assistants"
                ],
                metadatas=[
                    {"importance": 0.8, "topic": "programming"},
                    {"importance": 0.9, "topic": "ml"},
                    {"importance": 0.7, "topic": "voice"}
                ]
            )
            print("✅ Documents added successfully")
            
            # Test 4: Querying documents
            print("\n🔍 Testing document querying...")
            results = collection.query(
                query_texts=["programming languages"],
                n_results=2,
                include=["documents", "metadatas", "distances"]
            )
            print(f"✅ Query returned {len(results['documents'][0])} results")
            
            # Test 5: Collection info
            print("\n📊 Testing collection info...")
            count = collection.count()
            print(f"✅ Collection contains {count} documents")
            
            # Test 6: Collection listing
            print("\n📋 Testing collection listing...")
            collections = client.list_collections()
            print(f"✅ Found {len(collections)} collections")
            
            # Test 7: Memory system integration test
            print("\n🧠 Testing memory system integration...")
            from src.coda.components.memory.long_term import LongTermMemory
            from src.coda.components.memory.models import LongTermMemoryConfig
            
            # Create test config
            config = LongTermMemoryConfig(
                storage_path=str(Path(temp_dir) / "memory_test"),
                vector_db_type="chroma",
                embedding_model="all-MiniLM-L6-v2",
                max_memories=100
            )
            
            # Initialize memory system
            memory = LongTermMemory(config)
            print("✅ Memory system initialized with ChromaDB 1.0.15")
            
            print("\n🎉 All ChromaDB 1.0.15 tests passed!")
            return True
            
        finally:
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            print(f"🧹 Cleaned up test directory: {temp_dir}")
            
    except ImportError as e:
        print(f"❌ ChromaDB import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ ChromaDB test failed: {e}")
        return False

def test_chromadb_api_compatibility():
    """Test ChromaDB API compatibility with our codebase."""
    print("\n🔧 ChromaDB API Compatibility Test")
    print("=" * 50)
    
    try:
        import chromadb
        
        # Test the exact API calls used in our codebase
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Test PersistentClient (used in long_term.py)
            client = chromadb.PersistentClient(path=temp_dir)
            
            # Test get_or_create_collection with metadata (used in long_term.py)
            collection = client.get_or_create_collection(
                name="memories",
                metadata={"description": "Coda's long-term memories"},
                embedding_function=None
            )
            
            # Test add method with our data structure (used in _store_in_vector_db)
            collection.add(
                ids=["test_id"],
                embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5]],  # Mock embedding
                metadatas=[{"importance": 0.8, "topic": "test"}],
                documents=["Test memory content"]
            )
            
            # Test query method (used in _query_chroma)
            results = collection.query(
                query_embeddings=[[0.1, 0.2, 0.3, 0.4, 0.5]],
                n_results=1,
                include=["documents", "metadatas", "distances"]
            )
            
            # Verify result structure
            assert "documents" in results
            assert "metadatas" in results
            assert "distances" in results
            assert "ids" in results
            
            print("✅ All API calls compatible with ChromaDB 1.0.15")
            return True
            
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
            
    except Exception as e:
        print(f"❌ API compatibility test failed: {e}")
        return False

if __name__ == "__main__":
    success1 = test_chromadb_migration()
    success2 = test_chromadb_api_compatibility()
    
    if success1 and success2:
        print("\n🎉 ChromaDB 1.0.15 migration successful!")
        sys.exit(0)
    else:
        print("\n❌ ChromaDB migration issues detected")
        sys.exit(1)
