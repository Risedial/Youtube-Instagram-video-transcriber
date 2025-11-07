"""Proxy management system for YouTube Whisper Transcriber.

This module provides comprehensive proxy support with rotation, health monitoring,
anti-detection features, and kill switch functionality for both YouTube and Instagram.
"""

import time
import random
import logging
import threading
import requests
from typing import Optional, Dict, Any, List, Callable, Tuple
from urllib.parse import urlparse
from dataclasses import dataclass, field
from enum import Enum
import socket
import urllib3
from urllib3.exceptions import InsecureRequestWarning

# Disable SSL warnings for proxy connections
urllib3.disable_warnings(InsecureRequestWarning)

try:
    import socks
    SOCKS_AVAILABLE = True
except ImportError:
    SOCKS_AVAILABLE = False


class ProxyType(Enum):
    """Supported proxy types."""
    HTTP = "http"
    HTTPS = "https"
    SOCKS4 = "socks4"
    SOCKS5 = "socks5"


class ProxyStatus(Enum):
    """Proxy connection status."""
    UNKNOWN = "unknown"
    ACTIVE = "active"
    FAILED = "failed"
    BANNED = "banned"
    RATE_LIMITED = "rate_limited"


@dataclass
class ProxyConfig:
    """Proxy configuration data structure."""
    host: str
    port: int
    proxy_type: ProxyType = ProxyType.HTTP
    username: Optional[str] = None
    password: Optional[str] = None
    
    # Health tracking
    status: ProxyStatus = ProxyStatus.UNKNOWN
    success_count: int = 0
    failure_count: int = 0
    last_used: float = 0.0
    last_checked: float = 0.0
    response_time: float = 0.0
    
    # Usage tracking
    request_count: int = 0
    ban_count: int = 0
    rate_limit_count: int = 0

    def __post_init__(self):
        """Initialize timestamps."""
        if self.last_checked == 0.0:
            self.last_checked = time.time()
    
    @property
    def url(self) -> str:
        """Get proxy URL."""
        if self.username and self.password:
            return f"{self.proxy_type.value}://{self.username}:{self.password}@{self.host}:{self.port}"
        return f"{self.proxy_type.value}://{self.host}:{self.port}"
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0
    
    @property
    def is_healthy(self) -> bool:
        """Check if proxy is considered healthy."""
        return (self.status == ProxyStatus.ACTIVE and 
                self.success_rate >= 0.7 and 
                self.failure_count < 10)


@dataclass 
class AntiDetectionConfig:
    """Anti-detection configuration."""
    enable_random_delays: bool = True
    min_delay: int = 30
    max_delay: int = 120
    rotate_user_agents: bool = True
    randomize_headers: bool = True
    vary_request_patterns: bool = True
    
    # Instagram-specific
    instagram_max_requests_per_session: int = 10
    instagram_session_cooldown: int = 300
    instagram_preserve_cookies: bool = True


class ProxyRotationError(Exception):
    """Exception for proxy rotation issues."""
    pass


class KillSwitchError(Exception):
    """Exception for kill switch activation."""
    pass


class ProxyManager:
    """Comprehensive proxy management with rotation and health monitoring.
    
    Features:
    - Multiple proxy support with automatic rotation
    - Health monitoring and automatic failover
    - Anti-detection with random delays and header rotation
    - Kill switch functionality for connection failures
    - Platform-specific optimizations (Instagram, YouTube)
    - Request tracking and analytics
    """
    
    def __init__(self, settings=None):
        """Initialize proxy manager.
        
        Args:
            settings: Application settings instance
        """
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        
        # Proxy management
        self.proxies: List[ProxyConfig] = []
        self.current_proxy_index = 0
        self.rotation_lock = threading.Lock()
        
        # Health monitoring
        self.health_check_interval = 300  # 5 minutes
        self.health_check_thread: Optional[threading.Thread] = None
        self.health_check_running = False
        
        # Kill switch
        self.kill_switch_active = False
        self.kill_switch_thread: Optional[threading.Thread] = None
        self.connection_check_interval = 30
        
        # Anti-detection
        self.anti_detection = AntiDetectionConfig()
        self.last_request_time = 0.0
        self.current_user_agent_index = 0
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'proxy_rotations': 0,
            'kill_switch_activations': 0
        }
        
        # User agents for rotation
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.159 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:90.0) Gecko/20100101 Firefox/90.0',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:91.0) Gecko/20100101 Firefox/91.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15'
        ]
        
        # Platform-specific settings
        self.platform_configs = {
            'instagram': {
                'max_requests_per_session': 10,
                'session_cooldown': 300,
                'user_agent_consistency': True,
                'preserve_cookies': True
            },
            'youtube': {
                'user_agent_rotation': True,
                'aggressive_retry': True,
                'connection_pooling': True
            }
        }
        
        # Initialize from settings
        self._load_settings()
        
        self.logger.info(f"ProxyManager initialized with {len(self.proxies)} proxies")
    
    def _load_settings(self):
        """Load proxy settings from application settings."""
        if not self.settings:
            return
            
        # Load main proxy if configured
        if (hasattr(self.settings, '_settings') and 
            self.settings._settings.enable_proxy and 
            self.settings._settings.proxy_host):
            
            s = self.settings._settings
            proxy_config = ProxyConfig(
                host=s.proxy_host,
                port=s.proxy_port,
                proxy_type=ProxyType(s.proxy_type),
                username=s.proxy_username,
                password=s.proxy_password
            )
            self.proxies.append(proxy_config)
            
            # Load additional proxies from list
            for proxy_url in s.proxy_list:
                try:
                    proxy_config = self._parse_proxy_url(proxy_url)
                    self.proxies.append(proxy_config)
                except Exception as e:
                    self.logger.warning(f"Failed to parse proxy URL {proxy_url}: {e}")
            
            # Configure anti-detection
            self.anti_detection.enable_random_delays = s.enable_random_delays
            self.anti_detection.min_delay = s.min_request_delay
            self.anti_detection.max_delay = s.max_request_delay
            self.anti_detection.rotate_user_agents = s.rotate_user_agents
            self.anti_detection.randomize_headers = s.enable_header_randomization
            
            # Start health monitoring if proxies available
            if self.proxies and not self.health_check_running:
                self.start_health_monitoring()
                
            # Start kill switch if enabled
            if s.enable_kill_switch and not self.kill_switch_active:
                self.start_kill_switch()
    
    def _parse_proxy_url(self, proxy_url: str) -> ProxyConfig:
        """Parse proxy URL into ProxyConfig.
        
        Args:
            proxy_url: Proxy URL string
            
        Returns:
            ProxyConfig instance
        """
        parsed = urlparse(proxy_url)
        
        proxy_type = ProxyType.HTTP
        if parsed.scheme in ['socks4', 'socks5']:
            proxy_type = ProxyType(parsed.scheme)
        elif parsed.scheme == 'https':
            proxy_type = ProxyType.HTTPS
            
        return ProxyConfig(
            host=parsed.hostname,
            port=parsed.port or 8080,
            proxy_type=proxy_type,
            username=parsed.username,
            password=parsed.password
        )
    
    def get_current_proxy(self) -> Optional[ProxyConfig]:
        """Get current active proxy.
        
        Returns:
            Current proxy configuration or None
        """
        if not self.proxies:
            return None
            
        with self.rotation_lock:
            if self.current_proxy_index >= len(self.proxies):
                self.current_proxy_index = 0
            return self.proxies[self.current_proxy_index]
    
    def get_healthy_proxy(self) -> Optional[ProxyConfig]:
        """Get a healthy proxy, rotating if necessary.
        
        Returns:
            Healthy proxy configuration or None
        """
        if not self.proxies:
            return None
            
        with self.rotation_lock:
            # Try current proxy first
            current = self.get_current_proxy()
            if current and current.is_healthy:
                return current
            
            # Find next healthy proxy
            start_index = self.current_proxy_index
            for i in range(len(self.proxies)):
                index = (start_index + i) % len(self.proxies)
                proxy = self.proxies[index]
                
                if proxy.is_healthy:
                    self.current_proxy_index = index
                    self.stats['proxy_rotations'] += 1
                    self.logger.info(f"Rotated to proxy: {proxy.host}:{proxy.port}")
                    return proxy
            
            # No healthy proxies found, return current anyway
            self.logger.warning("No healthy proxies available, using current proxy")
            return current
    
    def rotate_proxy(self, reason: str = "manual") -> Optional[ProxyConfig]:
        """Manually rotate to next proxy.
        
        Args:
            reason: Reason for rotation
            
        Returns:
            New proxy configuration or None
        """
        if not self.proxies:
            return None
            
        with self.rotation_lock:
            old_index = self.current_proxy_index
            self.current_proxy_index = (self.current_proxy_index + 1) % len(self.proxies)
            new_proxy = self.proxies[self.current_proxy_index]
            
            self.stats['proxy_rotations'] += 1
            self.logger.info(f"Proxy rotation ({reason}): {old_index} -> {self.current_proxy_index}")
            
            return new_proxy
    
    def mark_proxy_failed(self, proxy: ProxyConfig, error: Exception):
        """Mark proxy as failed and update statistics.
        
        Args:
            proxy: Proxy that failed
            error: Exception that occurred
        """
        proxy.failure_count += 1
        proxy.status = ProxyStatus.FAILED
        
        # Check for specific error types
        if "banned" in str(error).lower() or "403" in str(error):
            proxy.status = ProxyStatus.BANNED
            proxy.ban_count += 1
        elif "429" in str(error) or "rate limit" in str(error).lower():
            proxy.status = ProxyStatus.RATE_LIMITED
            proxy.rate_limit_count += 1
        
        self.stats['failed_requests'] += 1
        self.logger.warning(f"Proxy {proxy.host}:{proxy.port} failed: {error}")
        
        # Auto-rotate if current proxy failed too many times
        if proxy.failure_count >= 5:
            self.rotate_proxy("too_many_failures")
    
    def mark_proxy_success(self, proxy: ProxyConfig, response_time: float):
        """Mark proxy as successful and update statistics.
        
        Args:
            proxy: Proxy that succeeded
            response_time: Response time in seconds
        """
        proxy.success_count += 1
        proxy.status = ProxyStatus.ACTIVE
        proxy.response_time = response_time
        proxy.last_used = time.time()
        proxy.request_count += 1
        
        self.stats['successful_requests'] += 1
        self.stats['total_requests'] += 1
    
    def check_proxy_health(self, proxy: ProxyConfig) -> bool:
        """Check if proxy is healthy by testing connection.
        
        Args:
            proxy: Proxy to test
            
        Returns:
            True if proxy is healthy
        """
        try:
            test_urls = [
                "https://httpbin.org/ip",
                "https://www.google.com",
                "https://www.cloudflare.com"
            ]
            
            proxies = {
                'http': proxy.url,
                'https': proxy.url
            }
            
            start_time = time.time()
            
            for url in test_urls:
                try:
                    response = requests.get(
                        url, 
                        proxies=proxies, 
                        timeout=10,
                        verify=False
                    )
                    
                    if response.status_code == 200:
                        response_time = time.time() - start_time
                        self.mark_proxy_success(proxy, response_time)
                        proxy.last_checked = time.time()
                        return True
                        
                except Exception as e:
                    continue
            
            # All test URLs failed
            self.mark_proxy_failed(proxy, Exception("Health check failed"))
            proxy.last_checked = time.time()
            return False
            
        except Exception as e:
            self.mark_proxy_failed(proxy, e)
            proxy.last_checked = time.time()
            return False
    
    def start_health_monitoring(self):
        """Start background health monitoring thread."""
        if self.health_check_running:
            return
            
        self.health_check_running = True
        self.health_check_thread = threading.Thread(
            target=self._health_monitor_loop,
            daemon=True
        )
        self.health_check_thread.start()
        self.logger.info("Started proxy health monitoring")
    
    def stop_health_monitoring(self):
        """Stop background health monitoring."""
        self.health_check_running = False
        if self.health_check_thread:
            self.health_check_thread.join(timeout=5)
        self.logger.info("Stopped proxy health monitoring")
    
    def _health_monitor_loop(self):
        """Background health monitoring loop."""
        while self.health_check_running:
            try:
                for proxy in self.proxies:
                    if not self.health_check_running:
                        break
                        
                    # Check if proxy needs health check
                    time_since_check = time.time() - proxy.last_checked
                    if time_since_check >= self.health_check_interval:
                        self.check_proxy_health(proxy)
                        
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in health monitor loop: {e}")
                time.sleep(60)
    
    def start_kill_switch(self):
        """Start kill switch monitoring."""
        if self.kill_switch_active:
            return
            
        self.kill_switch_active = True
        self.kill_switch_thread = threading.Thread(
            target=self._kill_switch_loop,
            daemon=True
        )
        self.kill_switch_thread.start()
        self.logger.info("Started kill switch monitoring")
    
    def stop_kill_switch(self):
        """Stop kill switch monitoring."""
        self.kill_switch_active = False
        if self.kill_switch_thread:
            self.kill_switch_thread.join(timeout=5)
        self.logger.info("Stopped kill switch monitoring")
    
    def _kill_switch_loop(self):
        """Background kill switch monitoring loop."""
        while self.kill_switch_active:
            try:
                # Check if any proxy is working
                healthy_proxies = [p for p in self.proxies if p.is_healthy]
                
                if not healthy_proxies and self.proxies:
                    self.logger.critical("Kill switch activated - no healthy proxies available")
                    self.stats['kill_switch_activations'] += 1
                    
                    # Trigger kill switch callbacks if registered
                    self._trigger_kill_switch()
                    
                time.sleep(self.connection_check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in kill switch loop: {e}")
                time.sleep(60)
    
    def _trigger_kill_switch(self):
        """Trigger kill switch - stop all network operations."""
        # This would integrate with the main application to stop downloads
        # For now, just log the event
        self.logger.critical("KILL SWITCH ACTIVATED - All network operations should be stopped")
    
    def apply_anti_detection_delay(self):
        """Apply random delay for anti-detection."""
        if not self.anti_detection.enable_random_delays:
            return
            
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        min_delay = self.anti_detection.min_delay
        required_delay = min_delay - time_since_last
        
        if required_delay > 0:
            # Add random component
            max_delay = self.anti_detection.max_delay
            random_delay = random.uniform(required_delay, min(max_delay, required_delay * 2))
            
            self.logger.debug(f"Anti-detection delay: {random_delay:.1f}s")
            time.sleep(random_delay)
        
        self.last_request_time = time.time()
    
    def get_randomized_headers(self, platform: str = "general") -> Dict[str, str]:
        """Get randomized headers for anti-detection.
        
        Args:
            platform: Platform name for specific headers
            
        Returns:
            Dictionary of HTTP headers
        """
        headers = {}
        
        # User agent rotation
        if self.anti_detection.rotate_user_agents:
            user_agent = random.choice(self.user_agents)
            headers['User-Agent'] = user_agent
        
        # Platform-specific headers
        if platform == "instagram":
            headers.update({
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1',
            })
        elif platform == "youtube":
            headers.update({
                'Accept': '*/*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept-Encoding': 'gzip, deflate, br',
                'Connection': 'keep-alive',
            })
        
        # Add random headers if enabled
        if self.anti_detection.randomize_headers:
            random_headers = [
                ('Cache-Control', random.choice(['no-cache', 'max-age=0'])),
                ('Pragma', 'no-cache'),
                ('Sec-Fetch-Dest', random.choice(['document', 'empty'])),
                ('Sec-Fetch-Mode', random.choice(['navigate', 'cors'])),
                ('Sec-Fetch-Site', random.choice(['none', 'same-origin'])),
            ]
            
            # Add 2-3 random headers
            for header_name, header_value in random.sample(random_headers, random.randint(2, 3)):
                headers[header_name] = header_value
        
        return headers
    
    def get_proxy_session(self, platform: str = "general") -> requests.Session:
        """Get configured requests session with proxy and anti-detection.
        
        Args:
            platform: Platform name for specific configuration
            
        Returns:
            Configured requests session
        """
        session = requests.Session()
        
        # Get healthy proxy
        proxy = self.get_healthy_proxy()
        if proxy:
            session.proxies = {
                'http': proxy.url,
                'https': proxy.url
            }
        
        # Set headers
        session.headers.update(self.get_randomized_headers(platform))
        
        # Configure session for platform
        if platform == "instagram":
            session.cookies.set_policy(None)  # Allow all cookies
        
        return session
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get proxy manager statistics.
        
        Returns:
            Dictionary with statistics
        """
        healthy_proxies = len([p for p in self.proxies if p.is_healthy])
        
        return {
            'total_proxies': len(self.proxies),
            'healthy_proxies': healthy_proxies,
            'current_proxy_index': self.current_proxy_index,
            'stats': self.stats.copy(),
            'proxy_details': [
                {
                    'host': p.host,
                    'port': p.port,
                    'type': p.proxy_type.value,
                    'status': p.status.value,
                    'success_rate': p.success_rate,
                    'request_count': p.request_count,
                    'response_time': p.response_time
                }
                for p in self.proxies
            ]
        }
    
    def shutdown(self):
        """Shutdown proxy manager and cleanup resources."""
        self.stop_health_monitoring()
        self.stop_kill_switch()
        self.logger.info("ProxyManager shutdown complete")