/**
 * WebSocket Voice Client for real-time voice communication.
 * 
 * This client provides a JavaScript interface for connecting to the
 * Coda voice WebSocket server, handling audio capture, streaming,
 * and real-time voice processing.
 */

class VoiceWebSocketClient {
    constructor(options = {}) {
        // Configuration
        this.serverUrl = options.serverUrl || 'ws://localhost:8765';
        this.authToken = options.authToken || null;
        this.userId = options.userId || null;
        this.autoReconnect = options.autoReconnect !== false;
        this.reconnectDelay = options.reconnectDelay || 3000;
        this.maxReconnectAttempts = options.maxReconnectAttempts || 5;
        
        // Audio configuration
        this.audioConfig = {
            sampleRate: options.sampleRate || 16000,
            channels: options.channels || 1,
            format: options.format || 'wav',
            chunkSizeMs: options.chunkSizeMs || 100,
            enableVAD: options.enableVAD !== false,
            vadThreshold: options.vadThreshold || 0.01
        };
        
        // State
        this.websocket = null;
        this.isConnected = false;
        this.isAuthenticated = false;
        this.conversationId = null;
        this.isRecording = false;
        this.reconnectAttempts = 0;
        
        // Audio components
        this.audioContext = null;
        this.mediaStream = null;
        this.audioProcessor = null;
        this.audioBuffer = [];
        
        // Event handlers
        this.eventHandlers = {
            connected: [],
            disconnected: [],
            authenticated: [],
            authFailed: [],
            voiceResponse: [],
            voiceStreamChunk: [],
            error: [],
            conversationStarted: [],
            conversationEnded: [],
            status: []
        };
        
        // Message queue for offline messages
        this.messageQueue = [];
        
        console.log('VoiceWebSocketClient initialized');
    }
    
    /**
     * Connect to the WebSocket server.
     */
    async connect() {
        try {
            console.log(`Connecting to ${this.serverUrl}...`);
            
            this.websocket = new WebSocket(this.serverUrl);
            
            this.websocket.onopen = () => {
                console.log('WebSocket connected');
                this.isConnected = true;
                this.reconnectAttempts = 0;
                this._emit('connected');
                
                // Send queued messages
                this._sendQueuedMessages();
                
                // Authenticate if credentials provided
                if (this.authToken && this.userId) {
                    this.authenticate(this.userId, this.authToken);
                }
            };
            
            this.websocket.onmessage = (event) => {
                this._handleMessage(JSON.parse(event.data));
            };
            
            this.websocket.onclose = (event) => {
                console.log('WebSocket disconnected:', event.code, event.reason);
                this.isConnected = false;
                this.isAuthenticated = false;
                this._emit('disconnected', { code: event.code, reason: event.reason });
                
                // Auto-reconnect if enabled
                if (this.autoReconnect && this.reconnectAttempts < this.maxReconnectAttempts) {
                    setTimeout(() => {
                        this.reconnectAttempts++;
                        console.log(`Reconnection attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}`);
                        this.connect();
                    }, this.reconnectDelay);
                }
            };
            
            this.websocket.onerror = (error) => {
                console.error('WebSocket error:', error);
                this._emit('error', { type: 'websocket', error });
            };
            
        } catch (error) {
            console.error('Failed to connect:', error);
            this._emit('error', { type: 'connection', error });
        }
    }
    
    /**
     * Disconnect from the WebSocket server.
     */
    disconnect() {
        if (this.websocket) {
            this.autoReconnect = false;
            this.websocket.close(1000, 'Client disconnect');
        }
        
        this._stopRecording();
    }
    
    /**
     * Authenticate with the server.
     */
    authenticate(userId, token) {
        this._sendMessage('auth', {
            user_id: userId,
            token: token
        });
    }
    
    /**
     * Start a new conversation.
     */
    async startConversation(conversationId = null) {
        if (!conversationId) {
            conversationId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        }
        
        this._sendMessage('conversation_start', {
            conversation_id: conversationId
        });
        
        return conversationId;
    }
    
    /**
     * End the current conversation.
     */
    endConversation() {
        if (this.conversationId) {
            this._sendMessage('conversation_end', {
                conversation_id: this.conversationId
            });
        }
    }
    
    /**
     * Start voice recording and streaming.
     */
    async startVoiceRecording(mode = 'adaptive') {
        try {
            // Initialize audio context
            if (!this.audioContext) {
                this.audioContext = new (window.AudioContext || window.webkitAudioContext)({
                    sampleRate: this.audioConfig.sampleRate
                });
            }
            
            // Get microphone access
            this.mediaStream = await navigator.mediaDevices.getUserMedia({
                audio: {
                    sampleRate: this.audioConfig.sampleRate,
                    channelCount: this.audioConfig.channels,
                    echoCancellation: true,
                    noiseSuppression: true,
                    autoGainControl: true
                }
            });
            
            // Create audio processor
            const source = this.audioContext.createMediaStreamSource(this.mediaStream);
            this.audioProcessor = this.audioContext.createScriptProcessor(4096, 1, 1);
            
            this.audioProcessor.onaudioprocess = (event) => {
                this._processAudioChunk(event.inputBuffer);
            };
            
            source.connect(this.audioProcessor);
            this.audioProcessor.connect(this.audioContext.destination);
            
            this.isRecording = true;
            
            // Send voice start message
            this._sendMessage('voice_start', {
                mode: mode,
                audio_config: this.audioConfig
            });
            
            console.log('Voice recording started');
            
        } catch (error) {
            console.error('Failed to start voice recording:', error);
            this._emit('error', { type: 'audio', error });
        }
    }
    
    /**
     * Stop voice recording.
     */
    stopVoiceRecording() {
        this._stopRecording();
        
        // Send any remaining audio buffer
        if (this.audioBuffer.length > 0) {
            this._sendAudioChunk(true); // is_final = true
        }
        
        // Send voice end message
        this._sendMessage('voice_end', {});
        
        console.log('Voice recording stopped');
    }
    
    /**
     * Send a text message for processing.
     */
    sendTextMessage(text, mode = 'adaptive') {
        this._sendMessage('voice_chunk', {
            text_content: text,
            audio_data: '',
            is_final: true,
            mode: mode
        });
    }
    
    /**
     * Add event listener.
     */
    on(event, handler) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].push(handler);
        }
    }
    
    /**
     * Remove event listener.
     */
    off(event, handler) {
        if (this.eventHandlers[event]) {
            const index = this.eventHandlers[event].indexOf(handler);
            if (index > -1) {
                this.eventHandlers[event].splice(index, 1);
            }
        }
    }
    
    /**
     * Get connection status.
     */
    getStatus() {
        return {
            isConnected: this.isConnected,
            isAuthenticated: this.isAuthenticated,
            conversationId: this.conversationId,
            isRecording: this.isRecording,
            reconnectAttempts: this.reconnectAttempts
        };
    }
    
    // Private methods
    
    _handleMessage(message) {
        console.log('Received message:', message.type);
        
        switch (message.type) {
            case 'connect':
                // Connection confirmation
                break;
                
            case 'auth_success':
                this.isAuthenticated = true;
                this._emit('authenticated', message.data);
                break;
                
            case 'auth_failed':
                this._emit('authFailed', message.data);
                break;
                
            case 'conversation_start':
                this.conversationId = message.data.conversation_id;
                this._emit('conversationStarted', message.data);
                break;
                
            case 'conversation_end':
                this.conversationId = null;
                this._emit('conversationEnded', message.data);
                break;
                
            case 'voice_response':
                this._handleVoiceResponse(message.data);
                break;
                
            case 'voice_stream_chunk':
                this._handleStreamChunk(message.data);
                break;
                
            case 'status':
                this._emit('status', message.data);
                break;
                
            case 'error':
                this._emit('error', { type: 'server', error: message.data });
                break;
                
            case 'pong':
                // Handle ping/pong for connection health
                break;
                
            default:
                console.warn('Unknown message type:', message.type);
        }
    }
    
    _handleVoiceResponse(data) {
        // Convert hex audio data back to bytes if present
        if (data.audio_data) {
            data.audio_data = this._hexToBytes(data.audio_data);
            this._playAudio(data.audio_data);
        }
        
        this._emit('voiceResponse', data);
    }
    
    _handleStreamChunk(data) {
        // Convert hex audio data back to bytes if present
        if (data.audio_data) {
            data.audio_data = this._hexToBytes(data.audio_data);
            this._playAudio(data.audio_data);
        }
        
        this._emit('voiceStreamChunk', data);
    }
    
    _processAudioChunk(inputBuffer) {
        if (!this.isRecording) return;
        
        // Get audio data
        const audioData = inputBuffer.getChannelData(0);
        
        // Apply voice activity detection if enabled
        if (this.audioConfig.enableVAD) {
            const rms = this._calculateRMS(audioData);
            if (rms < this.audioConfig.vadThreshold) {
                return; // Skip silent chunks
            }
        }
        
        // Add to buffer
        this.audioBuffer.push(new Float32Array(audioData));
        
        // Send chunk if buffer is large enough
        const chunkSamples = (this.audioConfig.chunkSizeMs * this.audioConfig.sampleRate) / 1000;
        const totalSamples = this.audioBuffer.reduce((sum, chunk) => sum + chunk.length, 0);
        
        if (totalSamples >= chunkSamples) {
            this._sendAudioChunk(false);
        }
    }
    
    _sendAudioChunk(isFinal = false) {
        if (this.audioBuffer.length === 0) return;
        
        // Combine buffer chunks
        const totalLength = this.audioBuffer.reduce((sum, chunk) => sum + chunk.length, 0);
        const combinedBuffer = new Float32Array(totalLength);
        
        let offset = 0;
        for (const chunk of this.audioBuffer) {
            combinedBuffer.set(chunk, offset);
            offset += chunk.length;
        }
        
        // Convert to 16-bit PCM
        const pcmData = new Int16Array(combinedBuffer.length);
        for (let i = 0; i < combinedBuffer.length; i++) {
            pcmData[i] = Math.max(-32768, Math.min(32767, combinedBuffer[i] * 32767));
        }
        
        // Convert to hex string
        const audioHex = Array.from(new Uint8Array(pcmData.buffer))
            .map(b => b.toString(16).padStart(2, '0'))
            .join('');
        
        // Send chunk
        this._sendMessage('voice_chunk', {
            audio_data: audioHex,
            is_final: isFinal,
            chunk_id: Date.now().toString()
        });
        
        // Clear buffer
        this.audioBuffer = [];
    }
    
    _sendMessage(type, data) {
        const message = {
            type: type,
            data: data,
            message_id: this._generateId(),
            timestamp: Date.now() / 1000,
            conversation_id: this.conversationId
        };
        
        if (this.isConnected && this.websocket.readyState === WebSocket.OPEN) {
            this.websocket.send(JSON.stringify(message));
        } else {
            // Queue message for later
            this.messageQueue.push(message);
        }
    }
    
    _sendQueuedMessages() {
        while (this.messageQueue.length > 0) {
            const message = this.messageQueue.shift();
            this.websocket.send(JSON.stringify(message));
        }
    }
    
    _stopRecording() {
        this.isRecording = false;
        
        if (this.audioProcessor) {
            this.audioProcessor.disconnect();
            this.audioProcessor = null;
        }
        
        if (this.mediaStream) {
            this.mediaStream.getTracks().forEach(track => track.stop());
            this.mediaStream = null;
        }
    }
    
    _playAudio(audioData) {
        // Create audio blob and play
        const blob = new Blob([audioData], { type: 'audio/wav' });
        const audioUrl = URL.createObjectURL(blob);
        const audio = new Audio(audioUrl);
        
        audio.play().catch(error => {
            console.error('Failed to play audio:', error);
        });
        
        // Clean up URL after playing
        audio.onended = () => {
            URL.revokeObjectURL(audioUrl);
        };
    }
    
    _calculateRMS(audioData) {
        let sum = 0;
        for (let i = 0; i < audioData.length; i++) {
            sum += audioData[i] * audioData[i];
        }
        return Math.sqrt(sum / audioData.length);
    }
    
    _hexToBytes(hex) {
        const bytes = new Uint8Array(hex.length / 2);
        for (let i = 0; i < hex.length; i += 2) {
            bytes[i / 2] = parseInt(hex.substr(i, 2), 16);
        }
        return bytes;
    }
    
    _generateId() {
        return Date.now().toString(36) + Math.random().toString(36).substr(2);
    }
    
    _emit(event, data) {
        if (this.eventHandlers[event]) {
            this.eventHandlers[event].forEach(handler => {
                try {
                    handler(data);
                } catch (error) {
                    console.error(`Error in ${event} handler:`, error);
                }
            });
        }
    }
}

// Export for use in different environments
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VoiceWebSocketClient;
} else if (typeof window !== 'undefined') {
    window.VoiceWebSocketClient = VoiceWebSocketClient;
}
