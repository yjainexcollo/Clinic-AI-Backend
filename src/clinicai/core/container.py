"""
Dependency injection container for Clinic-AI application.

This module provides a lightweight dependency injection container
for managing application dependencies and their lifecycle.
"""

from typing import Any, Callable, Dict, Generic, Optional, TypeVar

from .config import get_settings
from .exceptions import ConfigurationError

T = TypeVar("T")


class Container:
    """Lightweight dependency injection container."""

    def __init__(self) -> None:
        self._services: Dict[str, Any] = {}
        self._factories: Dict[str, Callable[[], Any]] = {}
        self._singletons: Dict[str, Any] = {}
        self._settings = get_settings()

    def register_singleton(self, name: str, instance: Any) -> None:
        """Register a singleton instance."""
        self._singletons[name] = instance

    def register_factory(self, name: str, factory: Callable[[], Any]) -> None:
        """Register a factory function."""
        self._factories[name] = factory

    def register_service(self, name: str, service: Any) -> None:
        """Register a service instance."""
        self._services[name] = service

    def get(self, name: str) -> Any:
        """Get a service by name."""
        # Check singletons firs
        if name in self._singletons:
            return self._singletons[name]

        # Check services
        if name in self._services:
            return self._services[name]

        # Check factories
        if name in self._factories:
            instance = self._factories[name]()
            # Cache as singleton if it's a factory
            self._singletons[name] = instance
            return instance

        raise ConfigurationError(f"Service '{name}' not found")

    def get_or_none(self, name: str) -> Optional[Any]:
        """Get a service by name, return None if not found."""
        try:
            return self.get(name)
        except ConfigurationError:
            return None

    def has(self, name: str) -> bool:
        """Check if a service is registered."""
        return name in self._services or name in self._factories or name in self._singletons

    def clear(self) -> None:
        """Clear all registered services."""
        self._services.clear()
        self._factories.clear()
        self._singletons.clear()


class ServiceProvider(Generic[T]):
    """Generic service provider for type-safe dependency injection."""

    def __init__(self, container: Container, service_name: str) -> None:
        self._container = container
        self._service_name = service_name

    def get(self) -> T:
        """Get the service instance."""
        return self._container.get(self._service_name)

    def get_or_none(self) -> Optional[T]:
        """Get the service instance or None if not found."""
        return self._container.get_or_none(self._service_name)


# Global container instance
_container = Container()


def get_container() -> Container:
    """Get the global container instance."""
    return _container


def register_singleton(name: str, instance: Any) -> None:
    """Register a singleton instance in the global container."""
    _container.register_singleton(name, instance)


def register_factory(name: str, factory: Callable[[], Any]) -> None:
    """Register a factory function in the global container."""
    _container.register_factory(name, factory)


def register_service(name: str, service: Any) -> None:
    """Register a service instance in the global container."""
    _container.register_service(name, service)


def get_service(name: str) -> Any:
    """Get a service from the global container."""
    return _container.get(name)


def get_service_or_none(name: str) -> Optional[Any]:
    """Get a service from the global container or None if not found."""
    return _container.get_or_none(name)


def has_service(name: str) -> bool:
    """Check if a service is registered in the global container."""
    return _container.has(name)


def create_provider(service_name: str) -> ServiceProvider:
    """Create a service provider for the given service name."""
    return ServiceProvider(_container, service_name)


# Common service names
class ServiceNames:
    """Common service names used throughout the application."""

    # Core services
    SETTINGS = "settings"
    CONTAINER = "container"

    # Database services
    DATABASE_SESSION = "database_session"
    PATIENT_REPOSITORY = "patient_repository"
    CONSULTATION_REPOSITORY = "consultation_repository"

    # External services
    OPENAI_CLIENT = "openai_client"
    DEEPGRAM_CLIENT = "deepgram_client"

    # Security services
    AUTH_SERVICE = "auth_service"
    JWT_SERVICE = "jwt_service"

    # Event services
    EVENT_BUS = "event_bus"


# Initialize core services
def initialize_core_services() -> None:
    """Initialize core services in the container."""
    # Register settings
    register_singleton(ServiceNames.SETTINGS, get_settings())

    # Register container itself
    register_singleton(ServiceNames.CONTAINER, _container)


# Auto-initialize core services
initialize_core_services()
