from dataclasses import dataclass
from functools import lru_cache

from backend.app.core.config import Settings, get_settings
from backend.app.db.session import get_session_factory
from backend.app.services.chat_service import ChatService
from backend.app.services.context_service import ContextService
from backend.app.services.llm_service import FakeLLMProvider, LLMProvider
from backend.app.services.nlp_service import NLPService, StubNLPService
from backend.app.services.prompt_service import PromptService
from backend.app.services.resource_service import ResourceService
from backend.app.services.risk_service import RiskService


@dataclass(frozen=True)
class AppContainer:
    settings: Settings
    session_factory: object
    context_service: ContextService
    prompt_service: PromptService
    nlp_service: NLPService
    risk_service: RiskService
    resource_service: ResourceService
    llm_provider: LLMProvider
    chat_service: ChatService


@lru_cache
def build_container() -> AppContainer:
    settings = get_settings()
    session_factory = get_session_factory()

    context_service = ContextService(session_factory=session_factory)
    prompt_service = PromptService(context_service=context_service)

    nlp_service: NLPService
    if settings.nlp_model_path:
        try:
            from backend.app.services.nlp_service import RealNLPService

            nlp_service = RealNLPService(model_path=settings.nlp_model_path)
        except Exception as exc:  # noqa: BLE001 — graceful degradation
            import logging

            logging.getLogger(__name__).warning(
                "RealNLPService failed to load (path=%r, error=%s); "
                "falling back to StubNLPService.",
                settings.nlp_model_path,
                exc,
            )
            nlp_service = StubNLPService()
    else:
        nlp_service = StubNLPService()
    risk_service = RiskService()
    resource_service = ResourceService(session_factory=session_factory)

    # Use DoubaoLLMProvider when credentials are configured; fall back to the
    # deterministic fake so the server stays runnable without API keys.
    llm_provider: LLMProvider
    if settings.doubao_api_key and settings.doubao_model:
        from backend.app.services.llm_service import DoubaoLLMProvider

        llm_provider = DoubaoLLMProvider(
            api_key=settings.doubao_api_key,
            model=settings.doubao_model,
            base_url=settings.doubao_base_url,
        )
    else:
        llm_provider = FakeLLMProvider()

    chat_service = ChatService(
        session_factory=session_factory,
        nlp_service=nlp_service,
        risk_service=risk_service,
        resource_service=resource_service,
        prompt_service=prompt_service,
        llm_provider=llm_provider,
    )

    return AppContainer(
        settings=settings,
        session_factory=session_factory,
        context_service=context_service,
        prompt_service=prompt_service,
        nlp_service=nlp_service,
        risk_service=risk_service,
        resource_service=resource_service,
        llm_provider=llm_provider,
        chat_service=chat_service,
    )


def get_container() -> AppContainer:
    return build_container()
