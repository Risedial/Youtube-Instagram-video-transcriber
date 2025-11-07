"""Network connectivity and retry logic utilities for YouTube Whisper Transcriber.

This module provides network connectivity testing, intelligent retry mechanisms,
rate limiting handling, and connection optimization for YouTube downloads.
"""

import time
import random
import logging
import threading
from typing import Optional, Callable, Dict, Any, List, Tuple
from urllib.parse import urlparse
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import socket

# Import proxy manager
try:
    from .proxy_manager import ProxyManager, ProxyConfig
    PROXY_MANAGER_AVAILABLE = True
except ImportError:
    PROXY_MANAGER_AVAILABLE = False


class NetworkError(Exception):
    """Custom exception for network-related errors."""
    pass


class ConnectionTester:
    """Network connectivity testing utilities."""
    
    def __init__(self):
        """Initialize connection tester."""
        self.logger = logging.getLogger(__name__)
        
    def test_internet_connection(self, timeout: int = 10) -> Tuple[bool, Dict[str, Any]]:
        """Test general internet connectivity.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            Tuple of (is_connected, connection_info)
        """
        test_urls = [
            "https://www.google.com",
            "https://www.cloudflare.com",
            "https://httpbin.org/get"
        ]
        
        connection_info = {
            'is_connected': False,
            'latency_ms': None,
            'successful_url': None,
            'errors': []
        }
        
        for url in test_urls:
            try:
                start_time = time.time()
                response = requests.head(url, timeout=timeout, allow_redirects=True)
                latency = (time.time() - start_time) * 1000
                
                if response.status_code == 200:
                    connection_info.update({
                        'is_connected': True,
                        'latency_ms': latency,
                        'successful_url': url
                    })
                    self.logger.debug(f"Internet connection verified via {url} ({latency:.1f}ms)")
                    return True, connection_info
                    
            except Exception as e:
                connection_info['errors'].append(f"{url}: {str(e)}")
                
        self.logger.warning("No internet connection detected")
        return False, connection_info
        
    def test_youtube_connectivity(self, timeout: int = 15) -> Tuple[bool, Dict[str, Any]]:
        """Test connectivity specifically to YouTube services.
        
        Args:
            timeout: Connection timeout in seconds
            
        Returns:
            Tuple of (is_accessible, youtube_info)
        """
        youtube_endpoints = [
            "https://www.youtube.com",
            "https://youtubei.googleapis.com/youtubei/v1/browse",
            "https://www.googleapis.com/youtube/v3/videos"
        ]
        
        youtube_info = {
            'is_accessible': False,
            'accessible_endpoints': [],
            'blocked_endpoints': [],
            'average_latency_ms': None,
            'rate_limited': False,
            'errors': []
        }
        
        latencies = []
        
        for endpoint in youtube_endpoints:
            try:
                start_time = time.time()
                response = requests.head(endpoint, timeout=timeout, allow_redirects=True)
                latency = (time.time() - start_time) * 1000
                latencies.append(latency)
                
                if response.status_code == 200:
                    youtube_info['accessible_endpoints'].append(endpoint)
                elif response.status_code == 429:
                    youtube_info['rate_limited'] = True
                    youtube_info['blocked_endpoints'].append(f"{endpoint} (rate limited)")
                else:
                    youtube_info['blocked_endpoints'].append(f"{endpoint} (HTTP {response.status_code})")
                    
            except requests.RequestException as e:
                youtube_info['errors'].append(f"{endpoint}: {str(e)}")
                youtube_info['blocked_endpoints'].append(f"{endpoint} (connection error)")
                
        if youtube_info['accessible_endpoints']:
            youtube_info['is_accessible'] = True
            youtube_info['average_latency_ms'] = sum(latencies) / len(latencies) if latencies else None
            
        return youtube_info['is_accessible'], youtube_info
        
    def test_dns_resolution(self, hostname: str = "www.youtube.com") -> Tuple[bool, Dict[str, Any]]:
        """Test DNS resolution for a hostname.
        
        Args:
            hostname: Hostname to resolve
            
        Returns:
            Tuple of (is_resolvable, dns_info)
        """
        dns_info = {
            'hostname': hostname,
            'is_resolvable': False,
            'ip_addresses': [],
            'resolution_time_ms': None,
            'error': None
        }
        
        try:
            start_time = time.time()
            ip_addresses = socket.gethostbyname_ex(hostname)[2]
            resolution_time = (time.time() - start_time) * 1000
            
            dns_info.update({
                'is_resolvable': True,
                'ip_addresses': ip_addresses,
                'resolution_time_ms': resolution_time
            })
            
            self.logger.debug(f"DNS resolution for {hostname}: {ip_addresses} ({resolution_time:.1f}ms)")
            
        except socket.gaierror as e:
            dns_info['error'] = f"DNS resolution failed: {str(e)}"
            self.logger.error(dns_info['error'])
            
        return dns_info['is_resolvable'], dns_info


class RetryManager:
    """Intelligent retry logic for network operations."""
    
    def __init__(self, max_retries: int = 3, base_delay: float = 1.0, 
                 max_delay: float = 60.0, backoff_factor: float = 2.0):
        """Initialize retry manager.
        
        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay between retries in seconds
            max_delay: Maximum delay between retries in seconds  
            backoff_factor: Exponential backoff multiplier
        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.logger = logging.getLogger(__name__)
        
    def should_retry(self, exception: Exception, attempt: int) -> bool:
        """Determine if operation should be retried.
        
        Args:
            exception: Exception that occurred
            attempt: Current attempt number (0-based)
            
        Returns:
            True if should retry
        """
        if attempt >= self.max_retries:
            return False
            
        # Retry on specific network errors
        retryable_errors = (
            requests.exceptions.ConnectionError,
            requests.exceptions.Timeout,
            requests.exceptions.HTTPError,
            socket.gaierror,
            socket.timeout
        )
        
        if isinstance(exception, retryable_errors):
            return True
            
        # Check HTTP status codes that are retryable
        if isinstance(exception, requests.exceptions.HTTPError):
            if hasattr(exception, 'response') and exception.response:
                status_code = exception.response.status_code
                # Retry on server errors and rate limiting
                if status_code in [429, 500, 502, 503, 504]:
                    return True
                    
        return False
        
    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay before next retry.
        
        Args:
            attempt: Current attempt number (0-based)
            
        Returns:
            Delay in seconds
        """
        # Exponential backoff with jitter
        delay = self.base_delay * (self.backoff_factor ** attempt)
        delay = min(delay, self.max_delay)
        
        # Add jitter to prevent thundering herd
        jitter = random.uniform(0.1, 0.5) * delay
        return delay + jitter
        
    def retry_operation(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with retry logic.
        
        Args:
            operation: Function to execute
            *args: Arguments for operation
            **kwargs: Keyword arguments for operation
            
        Returns:
            Operation result
            
        Raises:
            Exception: Last exception if all retries failed
        """
        last_exception = None
        
        for attempt in range(self.max_retries + 1):
            try:
                result = operation(*args, **kwargs)
                if attempt > 0:
                    self.logger.info(f"Operation succeeded after {attempt} retries")
                return result
                
            except Exception as e:
                last_exception = e
                
                if not self.should_retry(e, attempt):
                    self.logger.error(f"Operation failed after {attempt} attempts: {str(e)}")
                    raise e
                    
                if attempt < self.max_retries:
                    delay = self.calculate_delay(attempt)
                    self.logger.warning(
                        f"Operation failed (attempt {attempt + 1}/{self.max_retries + 1}): {str(e)}. "
                        f"Retrying in {delay:.1f}s"
                    )
                    time.sleep(delay)
                    
        # All retries exhausted
        self.logger.error(f"Operation failed after {self.max_retries + 1} attempts")
        raise last_exception


class RateLimitHandler:
    """Handle rate limiting from YouTube and other services."""
    
    def __init__(self, requests_per_minute: int = 30, burst_limit: int = 10):
        """Initialize rate limit handler.
        
        Args:
            requests_per_minute: Maximum requests per minute
            burst_limit: Maximum burst requests
        """
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.request_times: List[float] = []
        self.lock = threading.Lock()
        self.logger = logging.getLogger(__name__)
        
    def wait_if_needed(self) -> float:
        """Wait if rate limit would be exceeded.
        
        Returns:
            Wait time in seconds (0 if no wait needed)
        """
        with self.lock:
            current_time = time.time()
            
            # Remove requests older than 1 minute
            self.request_times = [t for t in self.request_times if current_time - t < 60]
            
            # Check if we need to wait
            wait_time = 0
            
            # Check burst limit
            recent_requests = [t for t in self.request_times if current_time - t < 10]  # Last 10 seconds
            if len(recent_requests) >= self.burst_limit:
                wait_time = max(wait_time, 10 - (current_time - recent_requests[0]))
                
            # Check per-minute limit
            if len(self.request_times) >= self.requests_per_minute:
                oldest_request = min(self.request_times)
                wait_time = max(wait_time, 60 - (current_time - oldest_request))
                
            if wait_time > 0:
                self.logger.info(f"Rate limit reached, waiting {wait_time:.1f}s")
                time.sleep(wait_time)
                current_time = time.time()
                
            # Record this request
            self.request_times.append(current_time)
            
            return wait_time
            
    def handle_rate_limit_response(self, response: requests.Response) -> bool:
        """Handle rate limit response from server.
        
        Args:
            response: HTTP response
            
        Returns:
            True if rate limit was handled
        """
        if response.status_code == 429:
            # Check for Retry-After header
            retry_after = response.headers.get('Retry-After')
            if retry_after:
                try:
                    wait_time = int(retry_after)
                    self.logger.warning(f"Server requested {wait_time}s wait due to rate limiting")
                    time.sleep(wait_time)
                    return True
                except ValueError:
                    pass
                    
            # Default wait for rate limiting
            wait_time = min(60, len(self.request_times) * 2)
            self.logger.warning(f"Rate limited by server, waiting {wait_time}s")
            time.sleep(wait_time)
            return True
            
        return False


class OptimizedHTTPSession:
    """HTTP session optimized for YouTube downloads with proxy support."""
    
    def __init__(self, max_retries: int = 3, timeout: int = 30, proxy_manager: Optional['ProxyManager'] = None):
        """Initialize optimized HTTP session.
        
        Args:
            max_retries: Maximum retry attempts
            timeout: Request timeout in seconds
            proxy_manager: ProxyManager instance for proxy support
        """
        self.session = requests.Session()
        self.timeout = timeout
        self.proxy_manager = proxy_manager
        self.logger = logging.getLogger(__name__)
        
        # Configure retry strategy
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["HEAD", "GET", "OPTIONS"]
        )
        
        # Mount adapter with retry strategy
        adapter = HTTPAdapter(max_retries=retry_strategy, pool_connections=10, pool_maxsize=10)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set optimized headers (will be updated by proxy manager if available)
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        
        # Configure proxy if available
        self._configure_proxy()
    
    def _configure_proxy(self):
        """Configure proxy settings for the session."""
        if not self.proxy_manager:
            return
            
        proxy = self.proxy_manager.get_healthy_proxy()
        if proxy:
            self.session.proxies.update({
                'http': proxy.url,
                'https': proxy.url
            })
            
            # Update headers with randomized ones
            randomized_headers = self.proxy_manager.get_randomized_headers("youtube")
            self.session.headers.update(randomized_headers)
            
            self.logger.debug(f"Configured proxy: {proxy.host}:{proxy.port}")
    
    def _handle_proxy_error(self, error: Exception, proxy: ProxyConfig):
        """Handle proxy-related errors.
        
        Args:
            error: Exception that occurred
            proxy: Proxy that failed
        """
        if self.proxy_manager:
            self.proxy_manager.mark_proxy_failed(proxy, error)
            
            # Try to get a new healthy proxy
            new_proxy = self.proxy_manager.get_healthy_proxy()
            if new_proxy and new_proxy != proxy:
                self._configure_proxy()
                return True
        return False
        
    def get(self, url: str, **kwargs) -> requests.Response:
        """Perform GET request with optimized settings and proxy support.
        
        Args:
            url: URL to request
            **kwargs: Additional arguments
            
        Returns:
            Response object
        """
        # Apply anti-detection delay if proxy manager is available
        if self.proxy_manager:
            self.proxy_manager.apply_anti_detection_delay()
        
        kwargs.setdefault('timeout', self.timeout)
        
        # Attempt request with proxy error handling
        max_proxy_retries = 3
        for attempt in range(max_proxy_retries):
            try:
                current_proxy = None
                if self.proxy_manager:
                    current_proxy = self.proxy_manager.get_current_proxy()
                
                start_time = time.time()
                response = self.session.get(url, **kwargs)
                
                # Mark proxy success if used
                if current_proxy and self.proxy_manager:
                    response_time = time.time() - start_time
                    self.proxy_manager.mark_proxy_success(current_proxy, response_time)
                
                return response
                
            except Exception as e:
                # Handle proxy-specific errors
                if current_proxy and self.proxy_manager:
                    if self._handle_proxy_error(e, current_proxy):
                        continue  # Retry with new proxy
                
                # Re-raise if not proxy-related or no more retries
                if attempt == max_proxy_retries - 1:
                    raise e
                    
        # This shouldn't be reached, but just in case
        return self.session.get(url, **kwargs)
        
    def head(self, url: str, **kwargs) -> requests.Response:
        """Perform HEAD request with optimized settings and proxy support.
        
        Args:
            url: URL to request
            **kwargs: Additional arguments
            
        Returns:
            Response object
        """
        # Apply anti-detection delay if proxy manager is available
        if self.proxy_manager:
            self.proxy_manager.apply_anti_detection_delay()
        
        kwargs.setdefault('timeout', self.timeout)
        
        # Attempt request with proxy error handling
        max_proxy_retries = 3
        for attempt in range(max_proxy_retries):
            try:
                current_proxy = None
                if self.proxy_manager:
                    current_proxy = self.proxy_manager.get_current_proxy()
                
                start_time = time.time()
                response = self.session.head(url, **kwargs)
                
                # Mark proxy success if used
                if current_proxy and self.proxy_manager:
                    response_time = time.time() - start_time
                    self.proxy_manager.mark_proxy_success(current_proxy, response_time)
                
                return response
                
            except Exception as e:
                # Handle proxy-specific errors
                if current_proxy and self.proxy_manager:
                    if self._handle_proxy_error(e, current_proxy):
                        continue  # Retry with new proxy
                
                # Re-raise if not proxy-related or no more retries
                if attempt == max_proxy_retries - 1:
                    raise e
                    
        # This shouldn't be reached, but just in case
        return self.session.head(url, **kwargs)
        
    def close(self):
        """Close the session."""
        self.session.close()


class NetworkHelper:
    """Comprehensive network helper with connectivity testing and retry logic."""
    
    def __init__(self, max_retries: int = 3, rate_limit: int = 30, proxy_manager: Optional['ProxyManager'] = None):
        """Initialize network helper.
        
        Args:
            max_retries: Maximum retry attempts
            rate_limit: Requests per minute limit
            proxy_manager: ProxyManager instance for proxy support
        """
        self.connection_tester = ConnectionTester()
        self.retry_manager = RetryManager(max_retries=max_retries)
        self.rate_limiter = RateLimitHandler(requests_per_minute=rate_limit)
        self.proxy_manager = proxy_manager
        self.session = OptimizedHTTPSession(max_retries=max_retries, proxy_manager=proxy_manager)
        self.logger = logging.getLogger(__name__)
        
    def check_connectivity(self) -> Dict[str, Any]:
        """Perform comprehensive connectivity check.
        
        Returns:
            Dictionary with connectivity information
        """
        connectivity_info = {
            'internet_connected': False,
            'youtube_accessible': False,
            'dns_working': False,
            'overall_status': 'disconnected',
            'details': {}
        }
        
        # Test internet connection
        internet_ok, internet_info = self.connection_tester.test_internet_connection()
        connectivity_info['internet_connected'] = internet_ok
        connectivity_info['details']['internet'] = internet_info
        
        if not internet_ok:
            connectivity_info['overall_status'] = 'no_internet'
            return connectivity_info
            
        # Test DNS resolution
        dns_ok, dns_info = self.connection_tester.test_dns_resolution()
        connectivity_info['dns_working'] = dns_ok
        connectivity_info['details']['dns'] = dns_info
        
        # Test YouTube connectivity
        youtube_ok, youtube_info = self.connection_tester.test_youtube_connectivity()
        connectivity_info['youtube_accessible'] = youtube_ok
        connectivity_info['details']['youtube'] = youtube_info
        
        # Determine overall status
        if youtube_ok:
            connectivity_info['overall_status'] = 'fully_connected'
        elif dns_ok:
            connectivity_info['overall_status'] = 'youtube_blocked'
        else:
            connectivity_info['overall_status'] = 'dns_issues'
            
        return connectivity_info
        
    def download_with_retry(self, url: str, **kwargs) -> requests.Response:
        """Download with automatic retry and rate limiting.
        
        Args:
            url: URL to download
            **kwargs: Additional arguments for request
            
        Returns:
            Response object
            
        Raises:
            NetworkError: If download fails after retries
        """
        def _download():
            # Apply rate limiting
            self.rate_limiter.wait_if_needed()
            
            # Perform request
            response = self.session.get(url, **kwargs)
            
            # Handle rate limiting
            if self.rate_limiter.handle_rate_limit_response(response):
                # Retry after rate limit handling
                response = self.session.get(url, **kwargs)
                
            response.raise_for_status()
            return response
            
        try:
            return self.retry_manager.retry_operation(_download)
        except Exception as e:
            raise NetworkError(f"Download failed for {url}: {str(e)}")
            
    def check_url_accessibility(self, url: str) -> Tuple[bool, Dict[str, Any]]:
        """Check if URL is accessible with retry logic.
        
        Args:
            url: URL to check
            
        Returns:
            Tuple of (is_accessible, check_info)
        """
        def _check():
            self.rate_limiter.wait_if_needed()
            response = self.session.head(url, allow_redirects=True)
            return response
            
        try:
            response = self.retry_manager.retry_operation(_check)
            return True, {
                'status_code': response.status_code,
                'accessible': response.status_code == 200,
                'redirect_url': response.url if response.url != url else None
            }
        except Exception as e:
            return False, {'error': str(e), 'accessible': False}
            
    def __del__(self):
        """Cleanup on destruction."""
        try:
            self.session.close()
        except Exception:
            pass


# Utility functions for common network operations
def test_internet_connection(timeout: int = 10) -> bool:
    """Quick test for internet connectivity.
    
    Args:
        timeout: Connection timeout in seconds
        
    Returns:
        True if internet is accessible
    """
    tester = ConnectionTester()
    is_connected, _ = tester.test_internet_connection(timeout)
    return is_connected


def test_youtube_access(timeout: int = 15) -> bool:
    """Quick test for YouTube accessibility.
    
    Args:
        timeout: Connection timeout in seconds
        
    Returns:
        True if YouTube is accessible
    """
    tester = ConnectionTester()
    is_accessible, _ = tester.test_youtube_connectivity(timeout)
    return is_accessible


def create_retry_session(max_retries: int = 3, proxy_manager: Optional['ProxyManager'] = None) -> OptimizedHTTPSession:
    """Create optimized HTTP session for downloads.
    
    Args:
        max_retries: Maximum retry attempts
        proxy_manager: ProxyManager instance for proxy support
        
    Returns:
        Optimized HTTP session
    """
    return OptimizedHTTPSession(max_retries=max_retries, proxy_manager=proxy_manager)