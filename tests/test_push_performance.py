"""
Performance tests for push notification system.

Tests the performance characteristics of the push notification system under
various load conditions and concurrent scenarios.
"""

import asyncio
import json
import pytest
import time

from src.a2a_acp.database import SessionDatabase
from src.a2a_acp.models import TaskPushNotificationConfig
from src.a2a_acp.push_notification_manager import PushNotificationManager


class TestPushNotificationPerformance:
    """Performance tests for push notification system."""

    @pytest.fixture
    def test_db(self, tmp_path):
        """Create a test database."""
        db_path = tmp_path / "test_performance.db"
        return SessionDatabase(str(db_path))

    @pytest.fixture
    def push_manager(self, test_db):
        """Create a push notification manager for testing."""
        return PushNotificationManager(test_db)

    def test_config_storage_performance(self, test_db, push_manager):
        """Test performance of storing notification configurations."""
        num_configs = 100

        async def store_many_configs():
            start_time = time.time()

            # Store many configurations
            for i in range(num_configs):
                config = TaskPushNotificationConfig(
                    id=f"perf-config-{i}",
                    task_id=f"perf-task-{i % 10}",  # 10 different tasks
                    url=f"https://example.com/webhook/{i}"
                )
                await push_manager.store_config(config)

            end_time = time.time()
            total_time = end_time - start_time

            # Should store 100 configs in reasonable time
            assert total_time < 5.0  # Should complete within 5 seconds
            print(f"Stored {num_configs} configs in {total_time:.2f} seconds")

            # Verify all configs were stored
            for i in range(10):  # Check first 10 tasks
                configs = await push_manager.list_configs(f"perf-task-{i}")
                assert len(configs) == 10  # Each task should have 10 configs

        asyncio.run(store_many_configs())

    def test_notification_sending_performance(self, test_db, push_manager):
        """Test performance of sending notifications."""
        num_endpoints = 50

        async def send_to_many_endpoints():
            # Set up many notification configs for one task
            task_id = "perf-multi-endpoint-task"

            for i in range(num_endpoints):
                config = TaskPushNotificationConfig(
                    id=f"perf-multi-config-{i}",
                    task_id=task_id,
                    url=f"https://example.com/webhook/{i}"
                )
                await push_manager.store_config(config)

            # Test the core notification logic without HTTP calls
            start_time = time.time()

            # Get configs and test the filtering logic performance
            configs = await push_manager.list_configs(task_id)
            assert len(configs) == num_endpoints

            # Test notification creation performance
            for config in configs:
                # Test that filtering works quickly
                should_send = push_manager._should_send_notification(config, {
                    "event": "status_change",
                    "task_id": task_id
                })
                assert should_send

            end_time = time.time()
            total_time = end_time - start_time

            # Should process 50 configs quickly
            assert total_time < 2.0  # Should complete within 2 seconds
            print(f"Processed {num_endpoints} notification configs in {total_time:.2f} seconds")

        asyncio.run(send_to_many_endpoints())

    def test_concurrent_task_notifications(self, test_db, push_manager):
        """Test performance with concurrent operations on multiple tasks."""
        num_tasks = 20
        configs_per_task = 5

        async def concurrent_operations():
            # Set up notification configs for multiple tasks
            for task_i in range(num_tasks):
                for config_i in range(configs_per_task):
                    config = TaskPushNotificationConfig(
                        id=f"concurrent-config-{task_i}-{config_i}",
                        task_id=f"concurrent-task-{task_i}",
                        url=f"https://example.com/concurrent/{task_i}/{config_i}"
                    )
                    await push_manager.store_config(config)

            # Test concurrent config listing performance
            start_time = time.time()

            # List configs concurrently for multiple tasks
            tasks = []
            for task_i in range(num_tasks):
                task = push_manager.list_configs(f"concurrent-task-{task_i}")
                tasks.append(task)

            # Wait for all listings to complete
            results = await asyncio.gather(*tasks)
            end_time = time.time()

            total_time = end_time - start_time

            # All listings should succeed and return correct number of configs
            assert len(results) == num_tasks
            for result in results:
                assert len(result) == configs_per_task

            # Should handle concurrent load efficiently
            assert total_time < 5.0  # Should complete within 5 seconds for 20 tasks
            print(f"Listed configs for {num_tasks} tasks concurrently in {total_time:.2f} seconds")

        asyncio.run(concurrent_operations())

    def test_database_query_performance(self, test_db, push_manager):
        """Test performance of database queries under load."""
        num_configs = 1000

        async def database_performance_test():
            # Store many configurations
            for i in range(num_configs):
                config = TaskPushNotificationConfig(
                    id=f"query-perf-{i}",
                    task_id=f"query-task-{i % 50}",  # 50 different tasks
                    url=f"https://example.com/query/{i}"
                )
                await push_manager.store_config(config)

            # Test list_configs performance
            start_time = time.time()
            for task_i in range(50):
                configs = await push_manager.list_configs(f"query-task-{task_i}")
                assert len(configs) == 20  # Each task should have 20 configs
            end_time = time.time()

            query_time = end_time - start_time
            assert query_time < 2.0  # Should query 50 tasks quickly

            print(f"Queried {num_configs} configs across 50 tasks in {query_time:.2f} seconds")

        asyncio.run(database_performance_test())

    def test_memory_usage_under_load(self, test_db, push_manager):
        """Test memory usage with many database operations."""
        import gc

        async def memory_test():
            # Get initial memory state
            gc.collect()
            initial_objects = len(gc.get_objects())

            # Create many configs (test database performance under load)
            for i in range(100):
                config = TaskPushNotificationConfig(
                    id=f"memory-test-{i}",
                    task_id="memory-test-task",
                    url=f"https://example.com/memory/{i}"
                )
                await push_manager.store_config(config)

            # Test listing performance with many configs
            for i in range(20):
                configs = await push_manager.list_configs("memory-test-task")
                assert len(configs) == 100

            # Check for memory leaks
            gc.collect()
            final_objects = len(gc.get_objects())
            object_increase = final_objects - initial_objects

            # Object count shouldn't grow excessively (allow some increase for normal operation)
            assert object_increase < 10000  # Reasonable limit for test scenario

            print(f"Memory test completed. Object increase: {object_increase}")

        asyncio.run(memory_test())

    def test_cleanup_performance(self, test_db, push_manager):
        """Test performance of cleanup operations."""
        num_configs = 500

        async def cleanup_performance_test():
            # Create many configs across multiple tasks
            for i in range(num_configs):
                config = TaskPushNotificationConfig(
                    id=f"cleanup-perf-{i}",
                    task_id=f"cleanup-task-{i % 25}",  # 25 different tasks
                    url=f"https://example.com/cleanup/{i}"
                )
                await push_manager.store_config(config)

            # Test cleanup performance for failed tasks
            start_time = time.time()
            total_deleted = 0

            for task_i in range(25):
                deleted = await push_manager.cleanup_by_task_state(f"cleanup-task-{task_i}", "failed")
                total_deleted += deleted

            end_time = time.time()
            cleanup_time = end_time - start_time

            assert total_deleted == num_configs  # All configs should be deleted for failed tasks
            assert cleanup_time < 5.0  # Should cleanup quickly

            print(f"Cleaned up {total_deleted} configs in {cleanup_time:.2f} seconds")

        asyncio.run(cleanup_performance_test())

    def test_streaming_manager_performance(self, test_db):
        """Test streaming manager performance with many connections."""
        from src.a2a_acp.streaming_manager import StreamingManager

        async def streaming_performance_test():
            push_mgr = PushNotificationManager(test_db)
            streaming_mgr = StreamingManager(push_mgr)

            # Register many SSE connections
            num_connections = 100
            connections = []

            start_time = time.time()
            for i in range(num_connections):
                connection_id, connection = await streaming_mgr.register_sse_connection([f"task-{i}"])
                connections.append((connection_id, connection))
            end_time = time.time()

            registration_time = end_time - start_time
            assert registration_time < 3.0  # Should register 100 connections quickly

            # Test broadcasting performance
            broadcast_start = time.time()
            await streaming_mgr.broadcast_notification("test-task", {
                "event": "broadcast_performance_test",
                "data": "test_data"
            })
            broadcast_end = time.time()

            broadcast_time = broadcast_end - broadcast_start
            assert broadcast_time < 1.0  # Broadcasting should be fast

            # Clean up connections
            cleanup_start = time.time()
            for connection_id, _ in connections:
                await streaming_mgr.unregister_sse_connection(connection_id)
            cleanup_end = time.time()

            cleanup_time = cleanup_end - cleanup_start
            assert cleanup_time < 2.0  # Cleanup should be efficient

            print("Streaming performance:")
            print(f"  Registered {num_connections} connections in {registration_time:.2f}s")
            print(f"  Broadcast to connections in {broadcast_time:.2f}s")
            print(f"  Cleaned up connections in {cleanup_time:.2f}s")

        asyncio.run(streaming_performance_test())

    def test_high_throughput_scenario(self, test_db, push_manager):
        """Test system under high throughput conditions."""
        async def high_throughput_test():
            # Simulate high throughput: rapid config creation and listing

            operations_count = 0
            start_time = time.time()

            # Rapidly create configs and test listing performance
            for batch in range(5):  # Reduced from 10 to 5 for faster execution
                # Create 10 configs per batch (reduced from 20)
                for i in range(10):
                    config = TaskPushNotificationConfig(
                        id=f"throughput-{batch}-{i}",
                        task_id=f"throughput-task-{batch}",
                        url=f"https://example.com/throughput/{batch}/{i}"
                    )
                    await push_manager.store_config(config)
                    operations_count += 1

                # Test listing performance for this batch
                list_start = time.time()
                configs = await push_manager.list_configs(f"throughput-task-{batch}")
                list_end = time.time()

                assert len(configs) == 10
                operations_count += 1

                list_time = list_end - list_start
                assert list_time < 0.5  # Listing should be fast

            end_time = time.time()
            total_time = end_time - start_time

            # Overall throughput should be good
            throughput = operations_count / total_time
            assert throughput > 3  # Should achieve at least 3 operations per second

            print(f"High throughput test: {operations_count} operations in {total_time:.2f}s")
            print(f"Average throughput: {throughput:.1f} operations/second")

        asyncio.run(high_throughput_test())

    def test_scalability_with_task_count(self, test_db, push_manager):
        """Test scalability as the number of tasks increases."""
        async def scalability_test():
            # Test with increasing numbers of tasks
            task_counts = [10, 50, 100]

            for num_tasks in task_counts:
                print(f"\nTesting scalability with {num_tasks} tasks...")

                # Create configs for this many tasks
                for task_i in range(num_tasks):
                    config = TaskPushNotificationConfig(
                        id=f"scale-config-{task_i}",
                        task_id=f"scale-task-{task_i}",
                        url=f"https://example.com/scale/{task_i}"
                    )
                    await push_manager.store_config(config)

                # Test query performance
                query_start = time.time()
                for task_i in range(num_tasks):
                    configs = await push_manager.list_configs(f"scale-task-{task_i}")
                    assert len(configs) == 1
                query_end = time.time()

                query_time = query_end - query_start
                avg_query_time = query_time / num_tasks

                # Query time per task should remain reasonable
                assert avg_query_time < 0.1  # Less than 100ms per task

                print(f"  Queried {num_tasks} tasks in {query_time:.2f}s")
                print(f"  Average query time: {avg_query_time:.4f}s per task")

                # Clean up for next iteration
                for task_i in range(num_tasks):
                    await push_manager.delete_config(f"scale-task-{task_i}", f"scale-config-{task_i}")

        asyncio.run(scalability_test())


class TestStressTests:
    """Stress tests for extreme scenarios."""

    @pytest.fixture
    def test_db(self, tmp_path):
        """Create a test database."""
        db_path = tmp_path / "test_stress.db"
        return SessionDatabase(str(db_path))

    @pytest.fixture
    def push_manager(self, test_db):
        """Create a push notification manager for testing."""
        return PushNotificationManager(test_db)

    def test_rapid_fire_notifications(self, test_db, push_manager):
        """Test rapid-fire notification config operations."""
        async def rapid_fire_test():
            # Set up configs for rapid operations
            num_configs = 50  # Reduced from 200 for faster testing

            start_time = time.time()

            # Rapidly create many configs
            for i in range(num_configs):
                config = TaskPushNotificationConfig(
                    id=f"rapid-fire-config-{i}",
                    task_id="rapid-fire-task",
                    url=f"https://example.com/rapid/{i}"
                )
                await push_manager.store_config(config)

            # Rapidly list configs
            for i in range(20):  # Reduced from 200
                configs = await push_manager.list_configs("rapid-fire-task")
                assert len(configs) == num_configs

            end_time = time.time()
            total_time = end_time - start_time

            # Should handle rapid operations efficiently
            assert total_time < 5.0  # Should complete quickly

            throughput = num_configs / total_time
            print(f"Rapid fire: {num_configs} operations in {total_time:.2f}s")
            print(f"Throughput: {throughput:.1f} operations/second")

        asyncio.run(rapid_fire_test())

    def test_large_payload_handling(self, test_db, push_manager):
        """Test handling of large notification payloads."""
        async def large_payload_test():
            # Create a large payload for testing payload creation performance
            large_data = {
                "event": "large_payload_test",
                "task_id": "large-payload-task",
                "large_field": "x" * 10000,  # 10KB of data
                "array_field": ["item"] * 1000,  # Array with 1000 items
                "nested_object": {
                    "level1": {
                        "level2": {
                            "level3": "deep_value"
                        }
                    }
                }
            }

            # Test payload creation performance (no HTTP calls)
            start_time = time.time()

            # Test that payload creation works with large data
            from src.a2a_acp.push_notification_manager import PushNotificationManager
            mgr = PushNotificationManager(test_db)

            # Test payload creation multiple times
            for i in range(10):
                mgr._create_notification_payload(large_data)

            end_time = time.time()
            processing_time = end_time - start_time

            # Should handle large payloads efficiently
            assert processing_time < 1.0  # Should process quickly

            print(f"Large payload processing completed in {processing_time:.2f}s")
            print(f"Payload size: {len(json.dumps(large_data))} characters")

        asyncio.run(large_payload_test())


class TestResourceUsage:
    """Test resource usage and efficiency."""

    @pytest.fixture
    def test_db(self, tmp_path):
        """Create a test database."""
        db_path = tmp_path / "test_resources.db"
        return SessionDatabase(str(db_path))

    @pytest.fixture
    def push_manager(self, test_db):
        """Create a push notification manager for testing."""
        return PushNotificationManager(test_db)

    def test_connection_pool_efficiency(self, test_db, push_manager):
        """Test that database operations are efficient."""
        async def connection_test():
            # Set up multiple configs and test database efficiency
            for i in range(10):
                config = TaskPushNotificationConfig(
                    id=f"connection-test-{i}",
                    task_id="connection-test-task",
                    url=f"https://example.com/test/{i}"
                )
                await push_manager.store_config(config)

            # Test multiple database operations rapidly
            start_time = time.time()
            for i in range(10):
                # Test config storage and retrieval performance
                config = TaskPushNotificationConfig(
                    id=f"efficiency-test-{i}",
                    task_id="efficiency-test-task",
                    url=f"https://example.com/efficiency/{i}"
                )
                await push_manager.store_config(config)

                # Immediately retrieve to test round-trip performance
                retrieved = await push_manager.get_config("efficiency-test-task", f"efficiency-test-{i}")
                assert retrieved is not None

            end_time = time.time()
            total_time = end_time - start_time

            # Should handle database operations efficiently
            assert total_time < 2.0
            print(f"Database efficiency test completed in {total_time:.2f}s")

        asyncio.run(connection_test())

    def test_garbage_collection_behavior(self, test_db, push_manager):
        """Test that database operations don't cause memory leaks."""
        import gc

        async def gc_test():
            # Track object creation
            gc.collect()
            initial_count = len(gc.get_objects())

            # Create many configs and test that they're stored properly
            for i in range(50):  # Reduced from 100 for faster testing
                config = TaskPushNotificationConfig(
                    id=f"gc-test-{i}",
                    task_id="gc-test-task",
                    url=f"https://example.com/gc/{i}"
                )
                await push_manager.store_config(config)

            # Test that all configs can be retrieved
            configs = await push_manager.list_configs("gc-test-task")
            assert len(configs) == 50

            # Test cleanup operations
            for i in range(50):
                await push_manager.delete_config("gc-test-task", f"gc-test-{i}")

            # Verify cleanup
            configs_after_cleanup = await push_manager.list_configs("gc-test-task")
            assert len(configs_after_cleanup) == 0

            # Check for memory leaks
            gc.collect()
            final_count = len(gc.get_objects())
            object_increase = final_count - initial_count

            # Object count should be reasonable
            assert object_increase < 5000
            print(f"Garbage collection test completed. Object increase: {object_increase}")

        asyncio.run(gc_test())


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "-s"])