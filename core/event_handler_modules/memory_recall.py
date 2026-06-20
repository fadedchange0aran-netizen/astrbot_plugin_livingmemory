"""
记忆召回模块
负责长期记忆的检索和注入
"""

import asyncio
from types import SimpleNamespace
from typing import TYPE_CHECKING

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent
from astrbot.api.platform import MessageType
from astrbot.api.provider import ProviderRequest
from astrbot.core.agent.message import TextPart

from ..utils import (
    OperationContext,
    format_memories_for_fake_tool_call,
    format_memories_for_fake_tool_call_deepseek_v4,
    format_memories_for_injection,
    get_owner_id,
    get_persona_id,
    resolve_owner_context,
)

if TYPE_CHECKING:
    from ..base.config_manager import ConfigManager
    from ..managers.conversation_manager import ConversationManager
    from ..managers.memory_engine import MemoryEngine
    from ..utils.injection_adapter import InjectionAdapter
    from .message_utils import MessageUtils


class MemoryRecall:
    """记忆召回类"""

    def __init__(
        self,
        context,
        config_manager: "ConfigManager",
        memory_engine: "MemoryEngine",
        conversation_manager: "ConversationManager",
        message_utils: "MessageUtils",
        injection_adapter: "InjectionAdapter",
    ):
        """
        初始化记忆召回模块

        Args:
            context: AstrBot上下文
            config_manager: 配置管理器
            memory_engine: 记忆引擎
            conversation_manager: 会话管理器
            message_utils: 消息处理工具
            injection_adapter: 注入适配器
        """
        self.context = context
        self.config_manager = config_manager
        self.memory_engine = memory_engine
        self.conversation_manager = conversation_manager
        self.message_utils = message_utils
        self.injection_adapter = injection_adapter

    @staticmethod
    def _normalize_identity_values(raw_values) -> set[str]:
        if raw_values is None:
            return set()
        if isinstance(raw_values, str):
            candidates = raw_values.replace("\n", ",").split(",")
        elif isinstance(raw_values, (list, tuple, set)):
            candidates = raw_values
        else:
            candidates = [raw_values]
        return {str(item).strip() for item in candidates if str(item).strip()}

    def _get_continuity_sender_ids(self, owner_context: dict[str, object]) -> set[str]:
        canonical_owner_id = str(owner_context.get("canonical_owner_id") or "").strip()
        sender_id = str(owner_context.get("sender_id") or "").strip()
        is_trusted_owner_sender = bool(owner_context.get("is_trusted_owner_sender"))

        if canonical_owner_id and is_trusted_owner_sender:
            trusted_ids = self._normalize_identity_values(
                self.config_manager.get("privacy_settings.trusted_sender_ids", [])
            )
            trusted_ids.add(canonical_owner_id)
            return trusted_ids

        if sender_id:
            return {sender_id}
        return set()

    @staticmethod
    def _truncate_text(text: str, max_chars: int) -> str:
        normalized = " ".join(str(text or "").split())
        if max_chars <= 0 or len(normalized) <= max_chars:
            return normalized
        if max_chars <= 1:
            return normalized[:max_chars]
        return normalized[: max_chars - 1].rstrip() + "…"

    def _build_session_continuity_excerpt(self, messages: list, max_chars: int) -> str:
        parts: list[str] = []
        for message in messages:
            content = self._truncate_text(getattr(message, "content", "") or "", max_chars)
            if not content:
                continue
            role = getattr(message, "role", "")
            role_label = "阿然" if role == "assistant" else "你" if role == "user" else "系统"
            parts.append(f"{role_label}: {content}")
        return " / ".join(parts)

    @staticmethod
    def _looks_like_stackchan_value(raw_value: object) -> bool:
        text = str(raw_value or "").strip().lower()
        if not text:
            return False
        return (
            text == "stackchan"
            or "pf::stackchan::" in text
            or text.startswith("stackchan-")
            or text.startswith("stackchan:")
            or "stackchan" in text
        )

    def _should_skip_recall_for_stackchan(
        self, event: AstrMessageEvent, req: ProviderRequest
    ) -> bool:
        if not self.config_manager.get("recall_engine.disable_for_stackchan", False):
            return False

        candidates = [
            getattr(req, "session_id", None),
            getattr(event, "session_id", None),
            getattr(event, "conversation_id", None),
            getattr(event, "unified_msg_origin", None),
            getattr(event, "client_platform", None),
            getattr(event, "platform_name", None),
            getattr(event, "platform", None),
            getattr(event, "source", None),
        ]

        get_platform_name = getattr(event, "get_platform_name", None)
        if callable(get_platform_name):
            try:
                candidates.append(get_platform_name())
            except Exception:
                pass

        for value in candidates:
            if self._looks_like_stackchan_value(value):
                return True
        return False

    async def _build_cross_platform_continuity_block(
        self,
        current_session_id: str,
        owner_context: dict[str, object],
    ) -> str:
        if not self.config_manager.get(
            "recall_engine.cross_platform_continuity_enabled", False
        ):
            return ""

        continuity_sender_ids = self._get_continuity_sender_ids(owner_context)
        if not continuity_sender_ids:
            return ""

        limit = max(
            1,
            int(
                self.config_manager.get(
                    "recall_engine.cross_platform_continuity_limit", 2
                )
            ),
        )
        max_chars = max(
            20,
            int(
                self.config_manager.get(
                    "recall_engine.cross_platform_continuity_max_chars", 120
                )
            ),
        )
        scan_limit = max(
            limit,
            int(
                self.config_manager.get(
                    "recall_engine.cross_platform_continuity_scan_limit", 8
                )
            ),
        )

        recent_sessions = await self.conversation_manager.get_recent_sessions(
            limit=scan_limit
        )
        continuity_lines: list[str] = []

        for session in recent_sessions:
            session_id = str(getattr(session, "session_id", "") or "").strip()
            if not session_id or session_id == current_session_id:
                continue

            participants = {
                str(item).strip()
                for item in getattr(session, "participants", []) or []
                if str(item).strip()
            }
            if not participants:
                continue
            if participants and not participants.intersection(continuity_sender_ids):
                continue

            messages = await self.conversation_manager.get_messages(
                session_id=session_id,
                limit=2,
                use_cache=False,
            )
            excerpt = self._build_session_continuity_excerpt(messages, max_chars)
            if not excerpt:
                continue

            platform = str(getattr(session, "platform", "") or "").strip() or "unknown"
            continuity_lines.append(f"- [{platform}] {excerpt}")
            if len(continuity_lines) >= limit:
                break

        if not continuity_lines:
            return ""

        return (
            "[跨窗连续性提示]\n"
            "当前对话仍然是主轴；以下只是你在其他入口刚刚发生的少量近况，可用于保持连续性，不要盖过当前窗口。\n"
            + "\n".join(continuity_lines)
        )

    async def _search_group_member_profiles(
        self,
        session_id: str,
        query: str,
        limit: int,
        sender_id: str | None = None,
    ) -> list[SimpleNamespace]:
        bm25_retriever = getattr(self.memory_engine, "bm25_retriever", None)
        if bm25_retriever is None:
            return []

        try:
            results = await bm25_retriever.search(
                query=query,
                limit=limit,
                session_id=session_id,
                persona_id=None,
                owner_id=None,
            )
        except Exception as exc:
            logger.warning(
                f"[{session_id}] 群成员档案 BM25 召回失败: {exc}",
                exc_info=True,
            )
            return []

        matched: list[SimpleNamespace] = []
        for result in results or []:
            metadata = getattr(result, "metadata", {}) or {}
            content = getattr(result, "content", "") or ""
            source_session = str(
                metadata.get("source_session") or metadata.get("session_id") or ""
            ).strip()
            if source_session != session_id:
                continue
            if sender_id and not self._group_profile_matches_sender(
                metadata, content, sender_id
            ):
                continue
            matched.append(
                SimpleNamespace(
                    doc_id=getattr(result, "doc_id", None),
                    content=content,
                    final_score=float(getattr(result, "score", 0.0) or 0.0),
                    metadata=metadata,
                )
            )
        return matched

    def _group_profile_matches_sender(
        self,
        metadata: dict[str, object],
        content: str,
        sender_id: str,
    ) -> bool:
        normalized_sender_id = str(sender_id or "").strip()
        if not normalized_sender_id:
            return False

        sender_ids = self._normalize_identity_values(
            metadata.get("member_sender_ids")
            or metadata.get("sender_ids")
            or metadata.get("sender_id")
            or metadata.get("qq")
        )
        if normalized_sender_id in sender_ids:
            return True
        return normalized_sender_id in str(content or "")

    async def _search_group_member_profile_for_sender(
        self,
        session_id: str,
        sender_id: str,
        limit: int,
    ) -> list[SimpleNamespace]:
        normalized_sender_id = str(sender_id or "").strip()
        if not normalized_sender_id:
            return []

        return await self._search_group_member_profiles(
            session_id=session_id,
            query=normalized_sender_id,
            limit=max(limit, 5),
            sender_id=normalized_sender_id,
        )

    async def handle_memory_recall(
        self, event: AstrMessageEvent, req: ProviderRequest
    ):
        """Query and inject long-term memory before LLM request"""
        try:
            session_id = event.unified_msg_origin
            logger.debug(f"[DEBUG-Recall] 获取到 unified_msg_origin: {session_id}")

            # 检测异常session_id
            if session_id and (
                "Error:" in session_id or "error:" in session_id.lower()
            ):
                logger.warning(
                    f"[{session_id}] 检测到异常的session_id，这可能导致记忆功能异常。"
                )

            async with OperationContext("记忆召回", session_id):
                prompt_text = getattr(req, "prompt", "")
                extra_parts = getattr(req, "extra_user_content_parts", [])
                has_prompt_text = isinstance(prompt_text, str) and bool(
                    prompt_text.strip()
                )
                has_extra_parts = bool(extra_parts)

                if not has_prompt_text and not has_extra_parts:
                    logger.debug(f"[{session_id}] 请求中无可用用户内容，跳过记忆召回")
                    return

                # 自动删除旧的注入记忆
                if self.config_manager.get("recall_engine.auto_remove_injected", True):
                    removed = self._remove_injected_memories_from_context(
                        req, session_id
                    )
                    removed += self._remove_fake_tool_call_from_context(req, session_id)
                    if removed > 0:
                        logger.info(
                            f"[{session_id}] 已清理 {removed} 处历史记忆注入片段"
                        )

                # 先提取用户消息（消息存储和召回都需要）
                actual_query = await self.message_utils.get_event_message_str(event)

                request_query = (
                    prompt_text.strip() if isinstance(prompt_text, str) else ""
                )

                # 存储用户消息（仅私聊），无论是否启用召回都需要
                is_group = event.get_message_type() == MessageType.GROUP_MESSAGE
                if not is_group and actual_query:
                    message_to_store = request_query
                    if not message_to_store:
                        message_to_store = (
                            await self.message_utils.extract_message_content(event, req)
                        )
                    if not message_to_store:
                        message_to_store = actual_query.strip()
                    await self.conversation_manager.add_message_from_event(
                        event=event,
                        role="user",
                        content=message_to_store,
                    )
                    await self.message_utils.enforce_message_limit(session_id)

                # 若 top_k <= 0，跳过记忆检索和注入，但上述清理和消息存储已执行
                if self._should_skip_recall_for_stackchan(event, req):
                    logger.info(f"[{session_id}] StackChan 轻量模式已启用，跳过记忆检索和注入")
                    return

                top_k = self.config_manager.get("recall_engine.top_k", 5)
                if top_k <= 0:
                    logger.info(
                        f"[{session_id}] top_k={top_k} <= 0，跳过记忆检索和注入"
                    )
                    return

                if not actual_query:
                    logger.warning(f"[{session_id}] 原始用户消息为空，跳过记忆召回")
                    return

                # 获取过滤配置
                filtering_config = self.config_manager.filtering_settings
                use_persona_filtering = filtering_config.get(
                    "use_persona_filtering", True
                )
                use_owner_filtering = filtering_config.get("use_owner_filtering", True)
                use_session_filtering = filtering_config.get(
                    "use_session_filtering", False
                )

                # 获取 persona_id，与 AstrBot 主流程保持一致的三级优先级：
                # 1. session_service_config（最高）
                # 2. req.conversation.persona_id（会话级）
                # 3. 全局默认人格（最低）
                # 注意：on_llm_request 钩子在 _ensure_persona_and_skills 之前触发，
                # 因此不能直接依赖 req.system_prompt 已注入人格，需自行走完整优先级。
                persona_id = await get_persona_id(self.context, event)
                owner_context = resolve_owner_context(self.config_manager, event)
                owner_id = get_owner_id(self.config_manager, event, purpose="recall")
                recalled_memories = []

                if use_owner_filtering and owner_id is None:
                    logger.info(
                        f"[{session_id}] 当前上下文不允许读取 owner 私有记忆，"
                        f"sender_id={owner_context.get('sender_id') or 'unknown'}"
                    )
                    if event.get_message_type() != MessageType.GROUP_MESSAGE:
                        return
                    group_sender_id = str(event.get_sender_id() or "").strip()
                    used_sender_profile = False
                    recalled_memories = (
                        await self._search_group_member_profile_for_sender(
                            session_id=session_id,
                            sender_id=group_sender_id,
                            limit=top_k,
                        )
                    )
                    if recalled_memories:
                        used_sender_profile = True
                        logger.info(
                            f"[{session_id}] 已按 sender_id={group_sender_id} 命中 "
                            f"{len(recalled_memories)} 条群成员档案记忆"
                        )
                    else:
                        recalled_memories = await self._search_group_member_profiles(
                            session_id=session_id,
                            query=actual_query,
                            limit=top_k,
                        )
                    if not recalled_memories:
                        return
                    if not used_sender_profile:
                        logger.info(
                            f"[{session_id}] 已按查询命中 {len(recalled_memories)} 条群成员档案记忆"
                        )

                if not recalled_memories:
                    continuity_block = await self._build_cross_platform_continuity_block(
                        session_id,
                        owner_context,
                    )
                    if continuity_block:
                        req.extra_user_content_parts.append(
                            TextPart(text=continuity_block).mark_as_temp()
                        )
                        logger.info(f"[{session_id}] 已注入跨窗连续性提示")

                    recall_owner_id = owner_id if use_owner_filtering else None
                    recall_session_id = session_id if use_session_filtering else None
                    recall_persona_id = persona_id if use_persona_filtering else None

                    # 使用原始用户输入作为召回关键字
                    query_for_search = actual_query

                    # 上下文扩展：拼接最近2轮对话作为查询，提升检索精准度
                    if self.config_manager.get(
                        "recall_engine.inject_with_recent_context", False
                    ):
                        try:
                            recent_messages = (
                                await self.conversation_manager.get_context(
                                    session_id, max_messages=5
                                )
                            )
                            if recent_messages and len(recent_messages) > 1:
                                # recent_messages 按 timestamp DESC 排列（最新在前）
                                # 跳过索引0（当前消息），取后续消息作为扩展上下文
                                context_parts = []
                                for msg in reversed(recent_messages[1:]):
                                    content = msg.get("content", "")
                                    if content and content.strip():
                                        context_parts.append(content.strip())
                                if context_parts:
                                    expanded = " | ".join(context_parts)
                                    query_for_search = expanded + " " + actual_query
                                    logger.info(
                                        f"[{session_id}] 上下文扩展查询: "
                                        f"{len(context_parts)}条历史消息 + 当前消息"
                                    )
                        except Exception as e:
                            logger.warning(f"[{session_id}] 获取上下文扩展失败: {e}")

                    # 执行记忆召回
                    logger.info(
                        f"[{session_id}] 开始记忆召回，查询='{query_for_search[:80]}...'"
                    )

                    recalled_memories = await self.memory_engine.search_memories(
                        query=query_for_search,
                        k=self.config_manager.get("recall_engine.top_k", 5),
                        owner_id=recall_owner_id,
                        session_id=recall_session_id,
                        persona_id=recall_persona_id,
                    )

                if recalled_memories:
                    logger.info(
                        f"[{session_id}] 检索到 {len(recalled_memories)} 条记忆"
                    )

                    # 格式化并注入记忆
                    memory_list = [
                        {
                            "id": getattr(mem, "doc_id", None),
                            "content": mem.content,
                            "score": mem.final_score,
                            "metadata": mem.metadata,
                            "timestamp": mem.metadata.get("create_time"),
                        }
                        for mem in recalled_memories
                    ]

                    # 输出详细记忆信息
                    for i, mem in enumerate(recalled_memories, 1):
                        logger.debug(
                            f"[{session_id}] 记忆 #{i}: 得分={mem.final_score:.3f}, "
                            f"重要性={mem.metadata.get('importance', 0.5):.2f}, "
                            f"内容={mem.content[:100]}..."
                        )

                    # 根据配置选择注入方式（含 Provider 兼容降级）
                    configured_method = self.config_manager.get(
                        "recall_engine.injection_method", "extra_user_content"
                    )
                    provider = None
                    if configured_method == "fake_tool_call":
                        provider = self.context.get_using_provider(session_id)
                    injection_method, fallback_reason = (
                        self.injection_adapter.resolve(provider, configured_method)
                    )
                    if fallback_reason:
                        logger.warning(
                            f"[{session_id}] 注入模式从 {configured_method} 降级为 "
                            f"{injection_method}: {fallback_reason}"
                        )

                    memory_str = format_memories_for_injection(memory_list)

                    if injection_method == "user_message_before":
                        req.prompt = memory_str + "\n\n" + (req.prompt or "")
                        logger.info(
                            f"[{session_id}] 成功向用户消息前注入 {len(recalled_memories)} 条记忆"
                        )
                    elif injection_method == "user_message_after":
                        req.prompt = (req.prompt or "") + "\n\n" + memory_str
                        logger.info(
                            f"[{session_id}] 成功向用户消息后注入 {len(recalled_memories)} 条记忆"
                        )
                    elif injection_method == "fake_tool_call":
                        fake_messages = format_memories_for_fake_tool_call(
                            memory_list,
                            query=actual_query,
                            k=self.config_manager.get("recall_engine.top_k", 5),
                            session_filtered=use_session_filtering,
                            persona_filtered=use_persona_filtering,
                        )
                        if fake_messages:
                            req.contexts.extend(fake_messages)
                            logger.info(
                                f"[{session_id}] 成功以伪造工具调用方式注入 "
                                f"{len(recalled_memories)} 条记忆"
                            )
                    elif injection_method == "fake_tool_call_deepseek_v4":
                        fake_replay = format_memories_for_fake_tool_call_deepseek_v4(
                            memory_list,
                            query=actual_query,
                            k=self.config_manager.get("recall_engine.top_k", 5),
                            session_filtered=use_session_filtering,
                            persona_filtered=use_persona_filtering,
                        )
                        if fake_replay:
                            req.prompt = fake_replay + "\n\n" + (req.prompt or "")
                            logger.info(
                                f"[{session_id}] 成功以 DeepSeek V4 兼容伪工具转录方式注入 "
                                f"{len(recalled_memories)} 条记忆"
                            )
                    else:
                        # extra_user_content（推荐）：追加到用户消息末尾，
                        # 不影响前缀缓存且 mark_as_temp 后不污染对话历史
                        req.extra_user_content_parts.append(
                            TextPart(text=memory_str).mark_as_temp()
                        )
                        logger.info(
                            f"[{session_id}] 成功向用户消息末尾注入 "
                            f"{len(recalled_memories)} 条记忆"
                        )
                else:
                    logger.info(f"[{session_id}] 未找到相关记忆")

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error(f"处理 on_llm_request 钩子时发生错误: {e}", exc_info=True)

    def _remove_injected_memories_from_context(
        self, req: ProviderRequest, session_id: str
    ) -> int:
        """从请求上下文中移除临时注入的记忆片段"""
        import re
        from ..base.constants import MEMORY_INJECTION_FOOTER, MEMORY_INJECTION_HEADER

        removed = 0

        # 清理 system_prompt（兼容旧版本注入残留）
        if hasattr(req, "system_prompt") and req.system_prompt:
            if isinstance(req.system_prompt, str):
                original_prompt = req.system_prompt
                if (
                    MEMORY_INJECTION_HEADER in original_prompt
                    and MEMORY_INJECTION_FOOTER in original_prompt
                ):
                    # 使用正则清理记忆片段
                    pattern = re.compile(
                        re.escape(MEMORY_INJECTION_HEADER)
                        + r".*?"
                        + re.escape(MEMORY_INJECTION_FOOTER),
                        re.DOTALL,
                    )
                    cleaned_prompt = pattern.sub("", original_prompt)
                    cleaned_prompt = re.sub(r"\n{3,}", "\n\n", cleaned_prompt).strip()
                    req.system_prompt = cleaned_prompt
                    if cleaned_prompt != original_prompt:
                        removed += 1

        # 清理 extra_user_content_parts（通过 is_temp 标记）
        parts_before = len(getattr(req, "extra_user_content_parts", []))
        if parts_before > 0:
            req.extra_user_content_parts = [
                part
                for part in req.extra_user_content_parts
                if not getattr(part, "is_temp", False)
            ]
            parts_after = len(req.extra_user_content_parts)
            removed += parts_before - parts_after

        return removed

    def _remove_fake_tool_call_from_context(
        self, req: ProviderRequest, session_id: str
    ) -> int:
        """从请求上下文中移除伪造的工具调用记忆（fake_tool_call 注入方式）

        识别并移除以 FAKE_TOOL_CALL_ID_PREFIX 为 ID 前缀的
        assistant(tool_calls) + tool(result) 消息对。
        """
        from ..base.constants import FAKE_TOOL_CALL_ID_PREFIX

        if not hasattr(req, "contexts") or not req.contexts:
            return 0

        removed = 0
        indices_to_remove: set[int] = set()
        fake_call_ids: set[str] = set()

        try:
            # 单轮扫描：同时收集伪造 assistant(tool_calls) 和对应 tool(result) 消息
            for i, msg in enumerate(req.contexts):
                if not isinstance(msg, dict):
                    continue
                role = msg.get("role")
                if role == "assistant" and msg.get("tool_calls"):
                    for tc in msg["tool_calls"]:
                        tc_id = (
                            tc.get("id", "")
                            if isinstance(tc, dict)
                            else getattr(tc, "id", "")
                        )
                        if tc_id.startswith(FAKE_TOOL_CALL_ID_PREFIX):
                            fake_call_ids.add(tc_id)
                            indices_to_remove.add(i)
                elif role == "tool":
                    tc_id = msg.get("tool_call_id", "")
                    if tc_id in fake_call_ids:
                        indices_to_remove.add(i)

            # 从后往前删除，避免索引偏移
            for i in sorted(indices_to_remove, reverse=True):
                req.contexts.pop(i)
                removed += 1

        except Exception:
            pass

        return removed
