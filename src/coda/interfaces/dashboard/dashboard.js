/**
 * Coda Dashboard - Real-time WebSocket Integration
 * 
 * This dashboard connects to the Coda WebSocket server to provide
 * real-time monitoring of component status, system health, and events.
 */

class CodaDashboard {
    constructor() {
        this.websocket = null;
        this.isConnected = false;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000;

        // Data storage
        this.components = new Map();
        this.systemStatus = null;
        this.eventHistory = [];
        this.performanceData = [];
        this.memoryData = [];

        // Charts
        this.performanceChart = null;
        this.memoryChart = null;

        // Event rate tracking
        this.eventCounts = [];
        this.eventRateInterval = null;

        // Chat functionality
        this.currentSessionId = null;
        this.messageHistory = [];
        this.isTyping = false;
        this.chatInput = null;
        this.sendButton = null;
        this.chatMessages = null;
        this.typingIndicator = null;

        this.init();
    }
    
    init() {
        console.log('🚀 Initializing Coda Dashboard...');

        // Initialize DOM elements
        this.initDOMElements();

        // Initialize charts
        this.initCharts();

        // Initialize chat interface
        this.initChatInterface();

        // Start event rate tracking
        this.startEventRateTracking();

        // Connect to WebSocket
        this.connect();

        // Set up periodic refresh
        setInterval(() => this.updateMetrics(), 1000);

        // Update welcome message time
        this.updateWelcomeTime();
    }
    
    connect() {
        const wsUrl = `ws://${window.location.hostname}:8765`;
        console.log(`🔌 Connecting to WebSocket: ${wsUrl}`);
        
        try {
            this.websocket = new WebSocket(wsUrl);
            
            this.websocket.onopen = (event) => {
                console.log('✅ WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this.updateConnectionStatus(true);
                this.addEventToLog('system', 'Connected to Coda WebSocket server');
            };
            
            this.websocket.onmessage = (event) => {
                try {
                    const data = JSON.parse(event.data);
                    this.handleWebSocketMessage(data);
                } catch (error) {
                    console.error('❌ Failed to parse WebSocket message:', error);
                }
            };
            
            this.websocket.onclose = (event) => {
                console.log('🔌 WebSocket disconnected');
                this.isConnected = false;
                this.updateConnectionStatus(false);
                this.addEventToLog('system', 'Disconnected from WebSocket server');
                this.attemptReconnect();
            };
            
            this.websocket.onerror = (error) => {
                console.error('❌ WebSocket error:', error);
                this.addEventToLog('error', 'WebSocket connection error');
            };
            
        } catch (error) {
            console.error('❌ Failed to create WebSocket connection:', error);
            this.updateConnectionStatus(false);
            this.attemptReconnect();
        }
    }
    
    attemptReconnect() {
        if (this.reconnectAttempts < this.maxReconnectAttempts) {
            this.reconnectAttempts++;
            const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
            
            console.log(`🔄 Attempting reconnect ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${delay}ms`);
            this.addEventToLog('system', `Reconnecting in ${delay/1000}s (attempt ${this.reconnectAttempts})`);
            
            setTimeout(() => this.connect(), delay);
        } else {
            console.error('❌ Max reconnection attempts reached');
            this.addEventToLog('error', 'Max reconnection attempts reached');
        }
    }
    
    handleWebSocketMessage(data) {
        console.log('📨 Received message:', data);
        
        // Track event for rate calculation
        this.eventCounts.push(Date.now());
        
        // Handle different event types
        switch (data.type) {
            case 'component_status':
                this.handleComponentStatus(data);
                break;
            case 'component_error':
                this.handleComponentError(data);
                break;
            case 'component_health':
                this.handleComponentHealth(data);
                break;
            case 'system_status':
                this.handleSystemStatus(data);
                break;
            case 'integration_metrics':
                this.handleIntegrationMetrics(data);
                break;
            case 'chat_response':
                this.handleChatResponse(data);
                break;
            case 'session_created':
                this.handleSessionCreated(data);
                break;
            case 'conversation_turn':
                this.handleConversationTurn(data);
                break;
            case 'llm_generation_start':
                this.handleLLMGenerationStart(data);
                break;
            case 'llm_generation_complete':
                this.handleLLMGenerationComplete(data);
                break;
            default:
                console.log('📋 Unknown event type:', data.type);
        }
        
        // Add to event history
        this.eventHistory.push({
            timestamp: new Date(),
            type: data.type,
            data: data
        });
        
        // Limit event history size
        if (this.eventHistory.length > 1000) {
            this.eventHistory = this.eventHistory.slice(-500);
        }
        
        // Update event log display
        this.updateEventLog();
    }
    
    handleComponentStatus(data) {
        const componentType = data.component_type;
        
        if (!this.components.has(componentType)) {
            this.components.set(componentType, {});
        }
        
        const component = this.components.get(componentType);
        component.state = data.state;
        component.initialization_order = data.initialization_order;
        component.dependencies = data.dependencies || [];
        component.lastUpdate = new Date();
        
        this.updateComponentsList();
        this.addEventToLog('component_status', `${componentType}: ${data.state}`);
    }
    
    handleComponentError(data) {
        const componentType = data.component_type;
        
        if (!this.components.has(componentType)) {
            this.components.set(componentType, {});
        }
        
        const component = this.components.get(componentType);
        component.state = 'failed';
        component.error_count = data.error_count;
        component.error_message = data.error_message;
        component.lastUpdate = new Date();
        
        this.updateComponentsList();
        this.addEventToLog('component_error', `${componentType}: ${data.error_message}`);
    }
    
    handleComponentHealth(data) {
        const componentType = data.component_type;
        
        if (!this.components.has(componentType)) {
            this.components.set(componentType, {});
        }
        
        const component = this.components.get(componentType);
        component.health_status = data.health_status;
        component.lastUpdate = new Date();
        
        this.addEventToLog('component_health', `${componentType}: Health updated`);
    }
    
    handleSystemStatus(data) {
        this.systemStatus = data;
        this.updateSystemOverview();
        this.addEventToLog('system_status', `System: ${data.integration_health}`);
    }
    
    handleIntegrationMetrics(data) {
        // Update performance metrics
        this.performanceData.push({
            timestamp: new Date(),
            events_processed: data.total_events_processed,
            active: data.is_active
        });
        
        // Limit data points
        if (this.performanceData.length > 50) {
            this.performanceData = this.performanceData.slice(-25);
        }
        
        this.updatePerformanceChart();
    }
    
    updateConnectionStatus(connected) {
        const indicator = document.getElementById('connectionIndicator');
        const status = document.getElementById('connectionStatus');
        
        if (connected) {
            indicator.classList.add('connected');
            status.textContent = 'Connected';
        } else {
            indicator.classList.remove('connected');
            status.textContent = 'Disconnected';
        }
    }
    
    updateComponentsList() {
        const componentList = document.getElementById('componentList');
        componentList.innerHTML = '';
        
        for (const [type, component] of this.components) {
            const listItem = document.createElement('li');
            listItem.className = 'component-item';
            
            const statusClass = this.getComponentStatusClass(component.state);
            
            listItem.innerHTML = `
                <span class="component-name">${type}</span>
                <div class="component-status ${statusClass}"></div>
            `;
            
            componentList.appendChild(listItem);
        }
    }
    
    getComponentStatusClass(state) {
        switch (state) {
            case 'initialized':
            case 'ready':
                return 'ready';
            case 'initializing':
                return 'initializing';
            case 'failed':
                return 'failed';
            case 'shutdown':
                return 'shutdown';
            default:
                return '';
        }
    }
    
    updateSystemOverview() {
        if (!this.systemStatus) return;
        
        document.getElementById('totalComponents').textContent = this.systemStatus.total_components;
        document.getElementById('readyComponents').textContent = this.systemStatus.ready_components;
        document.getElementById('failedComponents').textContent = this.systemStatus.failed_components;
        document.getElementById('systemHealth').textContent = this.systemStatus.integration_health;
    }
    
    updateEventLog() {
        const eventLog = document.getElementById('eventLog');
        const recentEvents = this.eventHistory.slice(-20).reverse();
        
        eventLog.innerHTML = recentEvents.map(event => {
            const time = event.timestamp.toLocaleTimeString();
            const type = event.type;
            const message = this.formatEventMessage(event);
            
            return `
                <div class="event-item">
                    <div class="event-time">${time}</div>
                    <div class="event-type ${type}">${type.toUpperCase()}</div>
                    <div class="event-message">${message}</div>
                </div>
            `;
        }).join('');
        
        // Auto-scroll to top (newest events)
        eventLog.scrollTop = 0;
    }
    
    formatEventMessage(event) {
        const data = event.data;
        
        switch (event.type) {
            case 'component_status':
                return `${data.component_type}: ${data.state}`;
            case 'component_error':
                return `${data.component_type}: ${data.error_message}`;
            case 'component_health':
                return `${data.component_type}: Health check`;
            case 'system_status':
                return `System health: ${data.integration_health}`;
            case 'integration_metrics':
                return `Events processed: ${data.total_events_processed}`;
            default:
                return JSON.stringify(data).substring(0, 100);
        }
    }
    
    addEventToLog(type, message) {
        const event = {
            timestamp: new Date(),
            type: type,
            data: { message: message }
        };
        
        this.eventHistory.push(event);
        this.updateEventLog();
    }
    
    startEventRateTracking() {
        this.eventRateInterval = setInterval(() => {
            const now = Date.now();
            const oneSecondAgo = now - 1000;
            
            // Count events in the last second
            const recentEvents = this.eventCounts.filter(timestamp => timestamp > oneSecondAgo);
            const eventRate = recentEvents.length;
            
            // Clean old events
            this.eventCounts = this.eventCounts.filter(timestamp => timestamp > now - 10000);
            
            // Update display
            document.getElementById('eventRate').textContent = eventRate;
        }, 1000);
    }
    
    initCharts() {
        // Performance Chart
        const performanceCtx = document.getElementById('performanceChart').getContext('2d');
        this.performanceChart = new Chart(performanceCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Events Processed',
                    data: [],
                    borderColor: '#4CAF50',
                    backgroundColor: 'rgba(76, 175, 80, 0.1)',
                    tension: 0.4
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    x: {
                        display: false
                    },
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(255, 255, 255, 0.1)'
                        },
                        ticks: {
                            color: 'rgba(255, 255, 255, 0.7)'
                        }
                    }
                }
            }
        });
        
        // Memory Chart
        const memoryCtx = document.getElementById('memoryChart').getContext('2d');
        this.memoryChart = new Chart(memoryCtx, {
            type: 'doughnut',
            data: {
                labels: ['Used', 'Available'],
                datasets: [{
                    data: [30, 70],
                    backgroundColor: ['#FF6384', '#36A2EB'],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    }
                }
            }
        });
    }
    
    updatePerformanceChart() {
        if (!this.performanceChart || this.performanceData.length === 0) return;
        
        const labels = this.performanceData.map(d => d.timestamp.toLocaleTimeString());
        const data = this.performanceData.map(d => d.events_processed);
        
        this.performanceChart.data.labels = labels;
        this.performanceChart.data.datasets[0].data = data;
        this.performanceChart.update('none');
    }
    
    updateMetrics() {
        // Simulate memory usage (in a real implementation, this would come from the server)
        const memoryUsage = Math.floor(Math.random() * 1000) + 500;
        document.getElementById('memoryUsage').textContent = `${memoryUsage} MB`;

        // Update memory chart
        const usedPercent = (memoryUsage / 2000) * 100;
        this.memoryChart.data.datasets[0].data = [usedPercent, 100 - usedPercent];
        this.memoryChart.update('none');
    }

    initDOMElements() {
        // Get chat interface elements
        this.chatInput = document.getElementById('chatInput');
        this.sendButton = document.getElementById('sendButton');
        this.chatMessages = document.getElementById('chatMessages');
        this.typingIndicator = document.getElementById('typingIndicator');
    }

    initChatInterface() {
        console.log('💬 Initializing chat interface...');

        if (!this.chatInput || !this.sendButton) {
            console.error('Chat interface elements not found');
            return;
        }

        // Set up event listeners
        this.chatInput.addEventListener('keydown', (e) => this.handleChatInputKeydown(e));
        this.chatInput.addEventListener('input', () => this.handleChatInputChange());
        this.sendButton.addEventListener('click', () => this.sendMessage());

        // Auto-resize textarea
        this.chatInput.addEventListener('input', () => this.autoResizeTextarea());

        console.log('✅ Chat interface initialized');
    }

    handleChatInputKeydown(event) {
        if (event.key === 'Enter' && !event.shiftKey) {
            event.preventDefault();
            this.sendMessage();
        }
    }

    handleChatInputChange() {
        // Update send button state
        const hasText = this.chatInput.value.trim().length > 0;
        this.sendButton.disabled = !hasText || this.isTyping || !this.isConnected;
    }

    autoResizeTextarea() {
        this.chatInput.style.height = 'auto';
        this.chatInput.style.height = Math.min(this.chatInput.scrollHeight, 100) + 'px';
    }

    async sendMessage() {
        const message = this.chatInput.value.trim();
        if (!message || this.isTyping || !this.isConnected) {
            return;
        }

        console.log('📤 Sending message:', message);

        // Add user message to chat
        this.addMessageToChat('user', message);

        // Clear input
        this.chatInput.value = '';
        this.chatInput.style.height = 'auto';
        this.handleChatInputChange();

        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Send message via WebSocket
            await this.sendChatMessage(message);
        } catch (error) {
            console.error('Failed to send message:', error);
            this.hideTypingIndicator();
            this.addMessageToChat('assistant', 'Sorry, I encountered an error processing your message. Please try again.');
        }
    }

    async sendChatMessage(message) {
        if (!this.websocket || this.websocket.readyState !== WebSocket.OPEN) {
            throw new Error('WebSocket not connected');
        }

        const chatMessage = {
            type: 'chat_message',
            data: {
                message: message,
                session_id: this.currentSessionId,
                timestamp: Date.now()
            },
            message_id: this.generateMessageId(),
            timestamp: Date.now() / 1000
        };

        this.websocket.send(JSON.stringify(chatMessage));
    }

    addMessageToChat(sender, content, timestamp = null) {
        const messageElement = document.createElement('div');
        messageElement.className = `message ${sender}`;

        const messageContent = document.createElement('div');
        messageContent.className = 'message-content';

        if (sender === 'assistant') {
            messageContent.innerHTML = `<i class="fas fa-robot" style="margin-right: 0.5rem; opacity: 0.7;"></i>${content}`;
        } else {
            messageContent.textContent = content;
        }

        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = timestamp || this.formatTime(new Date());

        messageElement.appendChild(messageContent);
        messageElement.appendChild(messageTime);

        // Insert before typing indicator
        this.chatMessages.insertBefore(messageElement, this.typingIndicator);

        // Scroll to bottom
        this.scrollChatToBottom();

        // Store in message history
        this.messageHistory.push({
            sender,
            content,
            timestamp: timestamp || new Date().toISOString()
        });
    }

    showTypingIndicator() {
        this.isTyping = true;
        this.typingIndicator.style.display = 'flex';
        this.scrollChatToBottom();
        this.handleChatInputChange();
    }

    hideTypingIndicator() {
        this.isTyping = false;
        this.typingIndicator.style.display = 'none';
        this.handleChatInputChange();
    }

    scrollChatToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTop = this.chatMessages.scrollHeight;
        }, 100);
    }

    formatTime(date) {
        return date.toLocaleTimeString('en-US', {
            hour: '2-digit',
            minute: '2-digit',
            hour12: false
        });
    }

    updateWelcomeTime() {
        const welcomeTimeElement = document.getElementById('welcomeTime');
        if (welcomeTimeElement) {
            welcomeTimeElement.textContent = this.formatTime(new Date());
        }
    }

    generateMessageId() {
        return 'msg_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
    }

    updateSessionStatus(sessionId) {
        this.currentSessionId = sessionId;
        const sessionStatus = document.getElementById('sessionStatus');
        const sessionIndicator = document.getElementById('sessionIndicator');

        if (sessionId) {
            sessionStatus.textContent = `Session: ${sessionId.substring(0, 8)}...`;
            sessionIndicator.style.background = '#4CAF50';
        } else {
            sessionStatus.textContent = 'No Session';
            sessionIndicator.style.background = '#ff4444';
        }
    }

    // Chat-specific event handlers
    handleChatResponse(data) {
        console.log('💬 Received chat response:', data);

        this.hideTypingIndicator();

        if (data.data && data.data.response) {
            const content = data.data.response.content || data.data.response;
            this.addMessageToChat('assistant', content);
        }

        // Update session if provided
        if (data.data && data.data.session_id) {
            this.updateSessionStatus(data.data.session_id);
        }
    }

    handleSessionCreated(data) {
        console.log('🆕 Session created:', data);

        if (data.data && data.data.session_id) {
            this.updateSessionStatus(data.data.session_id);
        }
    }

    handleConversationTurn(data) {
        console.log('🔄 Conversation turn:', data);

        // This event indicates a conversation is happening
        // We can use it to track conversation flow
    }

    handleLLMGenerationStart(data) {
        console.log('🤖 LLM generation started:', data);

        // Show typing indicator if not already shown
        if (!this.isTyping) {
            this.showTypingIndicator();
        }
    }

    handleLLMGenerationComplete(data) {
        console.log('✅ LLM generation complete:', data);

        // Hide typing indicator
        this.hideTypingIndicator();
    }
}

// Global functions for UI controls
function refreshDashboard() {
    console.log('🔄 Refreshing dashboard...');
    if (window.dashboard) {
        window.dashboard.addEventToLog('system', 'Dashboard refreshed');
        // Force reconnection if disconnected
        if (!window.dashboard.isConnected) {
            window.dashboard.connect();
        }
    }
}

function clearEventLog() {
    console.log('🗑️ Clearing event log...');
    if (window.dashboard) {
        window.dashboard.eventHistory = [];
        window.dashboard.updateEventLog();
        window.dashboard.addEventToLog('system', 'Event log cleared');
    }
}

function exportData() {
    console.log('💾 Exporting dashboard data...');
    if (window.dashboard) {
        const data = {
            timestamp: new Date().toISOString(),
            components: Object.fromEntries(window.dashboard.components),
            systemStatus: window.dashboard.systemStatus,
            eventHistory: window.dashboard.eventHistory.slice(-100) // Last 100 events
        };
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `coda-dashboard-${new Date().toISOString().split('T')[0]}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        window.dashboard.addEventToLog('system', 'Dashboard data exported');
    }
}

// Initialize dashboard when page loads
document.addEventListener('DOMContentLoaded', () => {
    window.dashboard = new CodaDashboard();
});
