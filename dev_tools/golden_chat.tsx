'use client'

import { useState } from 'react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

// Simulated blog posts data
const blogPosts = [
  { title: "The Future of AI in Conversation", excerpt: "Explore how AI is revolutionizing the way we communicate..." },
  { title: "5 Ways AI Can Boost Your Productivity", excerpt: "Discover the top AI-powered tools that can significantly increase your efficiency..." },
  { title: "Understanding Natural Language Processing", excerpt: "A deep dive into the technology behind AI's ability to understand and generate human language..." },
]

export default function Demo() {
  const [currentPage, setCurrentPage] = useState('home')
  const [messages, setMessages] = useState([
    { text: "Hello! How can I assist you today?", isBot: true },
  ])
  const [input, setInput] = useState("")

  const handleSend = () => {
    if (input.trim()) {
      setMessages([...messages, { text: input, isBot: false }])
      setInput("")
      // Simulated AI response
      setTimeout(() => {
        setMessages(prev => [...prev, { text: "I'm processing your request. This is a demo response.", isBot: true }])
      }, 1000)
    }
  }

  const renderHome = () => (
    <div className="text-center">
      <h1 className="text-4xl font-bold mb-4 text-yellow-800 metallic-gold">Welcome to AI Gold</h1>
      <p className="text-xl mb-8 text-yellow-900">Experience the future of AI-driven conversations</p>
      <Card className="max-w-md mx-auto bg-yellow-100 bg-opacity-50 backdrop-blur-sm border-yellow-300 shadow-gold">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-yellow-800">Start Your Journey</CardTitle>
        </CardHeader>
        <CardContent>
          <p className="mb-4 text-yellow-900">Unlock the power of AI with our cutting-edge chatbot technology.</p>
          <Button onClick={() => setCurrentPage('chat')} className="bg-gradient-to-r from-yellow-400 to-yellow-600 hover:from-yellow-500 hover:to-yellow-700 text-black font-bold py-2 px-4 rounded-full transition-all duration-300 transform hover:scale-105 shadow-gold">
            Try AI Chat Now
          </Button>
        </CardContent>
      </Card>
    </div>
  )

  const renderLogin = () => (
    <Card className="max-w-md mx-auto bg-yellow-100 bg-opacity-50 backdrop-blur-sm border-yellow-300 shadow-gold">
      <CardHeader>
        <CardTitle className="text-2xl font-bold text-yellow-800 metallic-gold">Login</CardTitle>
      </CardHeader>
      <CardContent>
        <form>
          <div className="mb-4">
            <Label htmlFor="email" className="text-yellow-900">Email</Label>
            <Input id="email" type="email" className="mt-1 bg-yellow-50 border-yellow-300 focus:border-yellow-500 focus:ring-yellow-500" />
          </div>
          <div className="mb-4">
            <Label htmlFor="password" className="text-yellow-900">Password</Label>
            <Input id="password" type="password" className="mt-1 bg-yellow-50 border-yellow-300 focus:border-yellow-500 focus:ring-yellow-500" />
          </div>
        </form>
      </CardContent>
      <CardFooter>
        <Button className="w-full bg-gradient-to-r from-yellow-400 to-yellow-600 hover:from-yellow-500 hover:to-yellow-700 text-black font-bold py-2 px-4 rounded-full transition-all duration-300 transform hover:scale-105 shadow-gold">
          Log In
        </Button>
      </CardFooter>
    </Card>
  )

  const renderBlog = () => (
    <div>
      <h1 className="text-3xl font-bold mb-6 text-yellow-800 metallic-gold">AI Gold Blog</h1>
      <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {blogPosts.map((post, index) => (
          <Card key={index} className="bg-yellow-100 bg-opacity-50 backdrop-blur-sm border-yellow-300 shadow-gold">
            <CardHeader>
              <CardTitle className="text-xl font-bold text-yellow-800">{post.title}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-yellow-900">{post.excerpt}</p>
            </CardContent>
          </Card>
        ))}
      </div>
    </div>
  )

  const renderChat = () => (
    <Card className="max-w-2xl mx-auto bg-yellow-100 bg-opacity-50 backdrop-blur-sm border-yellow-300 shadow-gold">
      <CardHeader>
        <CardTitle className="text-2xl font-bold text-yellow-800 metallic-gold">AI Gold Chat</CardTitle>
      </CardHeader>
      <CardContent className="h-96 overflow-y-auto">
        {messages.map((message, index) => (
          <div key={index} className={`mb-4 ${message.isBot ? 'text-left' : 'text-right'}`}>
            <span className={`inline-block p-2 rounded-lg ${message.isBot ? 'bg-yellow-200 text-yellow-900' : 'bg-yellow-500 text-white'}`}>
              {message.text}
            </span>
          </div>
        ))}
      </CardContent>
      <CardFooter>
        <div className="flex w-full">
          <Input 
            value={input} 
            onChange={(e) => setInput(e.target.value)} 
            placeholder="Type your message..." 
            className="flex-grow mr-2 bg-yellow-50 border-yellow-300 focus:border-yellow-500 focus:ring-yellow-500"
          />
          <Button onClick={handleSend} className="bg-gradient-to-r from-yellow-400 to-yellow-600 hover:from-yellow-500 hover:to-yellow-700 text-black font-bold py-2 px-4 rounded-full transition-all duration-300 transform hover:scale-105 shadow-gold">
            Send
          </Button>
        </div>
      </CardFooter>
    </Card>
  )

  return (
    <div className="min-h-screen bg-gradient-to-br from-yellow-200 via-yellow-300 to-yellow-400">
      <nav className="bg-black bg-opacity-20 backdrop-blur-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <span className="font-bold text-2xl text-yellow-300 metallic-gold cursor-pointer" onClick={() => setCurrentPage('home')}>AI Gold</span>
            </div>
            <div className="flex">
              <Button variant="link" className="text-yellow-300 hover:text-yellow-100" onClick={() => setCurrentPage('home')}>Home</Button>
              <Button variant="link" className="text-yellow-300 hover:text-yellow-100" onClick={() => setCurrentPage('blog')}>Blog</Button>
              <Button variant="link" className="text-yellow-300 hover:text-yellow-100" onClick={() => setCurrentPage('chat')}>Chat</Button>
              <Button variant="outline" className="ml-4 border-yellow-300 text-yellow-300 hover:bg-yellow-300 hover:text-black" onClick={() => setCurrentPage('login')}>Login</Button>
            </div>
          </div>
        </div>
      </nav>
      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {currentPage === 'home' && renderHome()}
        {currentPage === 'login' && renderLogin()}
        {currentPage === 'blog' && renderBlog()}
        {currentPage === 'chat' && renderChat()}
      </main>
    </div>
  )
}