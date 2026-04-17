# Railway Deployment Guide

Complete step-by-step guide to deploy your Farsi Transcriber web app to Railway.

## Prerequisites

1. **GitHub Account** - To connect your repository
2. **Railway Account** - Free signup at https://railway.app
3. **Git** - Already have this since you're using it

## Step 1: Prepare Your Code for Deployment

### 1.1 Make sure all files are committed

```bash
cd /home/user/whisper
git status
git add .
git commit -m "Ready for Railway deployment"
```

### 1.2 Push to GitHub

If you haven't already, push your fork to GitHub:

```bash
git push origin claude/review-repo-011CV3PVcA7ZSCTW2YquuMB8
```

---

## Step 2: Create Railway Account

1. Go to **https://railway.app**
2. Click **"Login with GitHub"**
3. Authorize Railway to access your GitHub account
4. You'll get **$5 monthly credit** for free ✅

---

## Step 3: Create Backend Service (Flask API)

### 3.1 Create a new project

1. In Railway dashboard, click **"Create New Project"**
2. Select **"GitHub Repo"**
3. Select your **whisper** repository
4. Railway will auto-detect it as Python
5. Configure:
   - **Root Directory:** `farsi_transcriber_web/backend`
   - **Start Command:** `gunicorn --workers 2 --bind 0.0.0.0:$PORT app:app`

### 3.2 Set environment variables

In the Railway dashboard for your backend:

1. Go to **Variables**
2. Add:
   ```
   FLASK_ENV=production
   FLASK_DEBUG=False
   PYTHONUNBUFFERED=1
   PORT=5000
   ```

### 3.3 Deploy

1. Click **Deploy**
2. Wait for deployment (takes 2-3 minutes)
3. Your backend URL will appear (e.g., `https://farsi-api-prod.railway.app`)
4. **Copy this URL** - you'll need it for the frontend

---

## Step 4: Create Frontend Service (React)

### 4.1 Create another service in same project

1. Go to your Railway project
2. Click **"Create New Service"**
3. Select **"GitHub Repo"**
4. Select your **whisper** repository
5. Configure:
   - **Root Directory:** `farsi_transcriber_web`
   - **Build Command:** `npm install && npm run build`
   - **Start Command:** `npm run preview`

### 4.2 Set environment variables

In the Railway dashboard for your frontend:

1. Go to **Variables**
2. Add:
   ```
   VITE_API_URL=https://your-backend-url-here.railway.app
   VITE_APP_NAME=Farsi Transcriber
   ```
   (Replace with your actual backend URL from Step 3.3)

### 4.3 Deploy

1. Click **Deploy**
2. Wait for deployment (2-5 minutes depending on npm install)
3. Your frontend URL will appear (e.g., `https://farsi-web-prod.railway.app`)

---

## Step 5: Configure Services to Communicate

### 5.1 Link backend to frontend

1. In Railway dashboard, select frontend service
2. Go to **Variables**
3. Update `VITE_API_URL` with your backend service domain
4. Deploy again

### 5.2 Test the connection

1. Open your frontend URL
2. Try to add a file and transcribe
3. Check browser console for any errors
4. If errors, check Railway logs (click service → Logs)

---

## Step 6: Monitor Your Deployment

### 6.1 View logs

In Railway dashboard:
- Click your service
- Go to **Logs** tab
- See real-time logs as users interact with your app

### 6.2 Check health

```bash
# Check if backend is running
curl https://your-backend-url.railway.app/health

# Should return:
# {"status": "healthy", "model_loaded": true, "environment": "production"}
```

### 6.3 Monitor usage

- Railway dashboard shows RAM, CPU, bandwidth usage
- Your $5 credit should last 1-3 months for personal use

---

## Step 7: Custom Domain (Optional)

If you want a custom domain like `farsi.yourdomain.com`:

1. Buy a domain on GoDaddy, Namecheap, etc.
2. In Railway dashboard → Your app → Settings → Domains
3. Add custom domain
4. Update DNS records at your domain provider
5. Railway will handle SSL certificate automatically

---

## Troubleshooting

### Issue: Backend showing error "Model not loaded"

**Solution:** First transcription loads the 769MB model (takes 1-2 min). Wait and try again.

### Issue: Frontend can't reach backend

**Solution:**
1. Check backend URL is correct in frontend variables
2. Backend must be running (check Railway logs)
3. CORS should be enabled (already configured)

### Issue: Build fails

**Solution:**
1. Check Railway build logs for errors
2. Ensure `package.json` has all required dependencies
3. Run locally first: `npm install && npm run build`

### Issue: App runs slow

**Solution:**
1. You're on free tier with limited resources
2. Upgrade to paid tier ($5/month) for better performance
3. Or wait for model to cache (subsequent transcriptions are fast)

### Issue: Out of memory

**Solution:**
1. Free tier has limited RAM
2. Close unused tabs/apps
3. Use smaller Whisper model (edit backend to use 'small' instead of 'medium')

---

## Next Steps: Custom Domain Setup

Once stable, add your custom domain:

1. Purchase domain
2. Railway → Settings → Domains → Add Domain
3. Update DNS CNAME records
4. Railway auto-generates SSL certificate

---

## Cost Breakdown

### Free Tier ($5/month credit)

- ✅ 500 build minutes/month
- ✅ 100 GB bandwidth/month
- ✅ 6,000 compute unit hours
- ✅ More than enough for personal use

### Your app will cost:

- **Backend (Flask):** ~$1-2/month
- **Frontend (React):** ~$0.50/month
- **Total:** ~$2/month (with free credit covering 2-3 months)

---

## Useful Commands

### Check if Railway CLI is installed

```bash
railway --version
```

### Install Railway CLI

```bash
npm i -g @railway/cli
```

### Deploy from command line

```bash
railway up
```

### View logs

```bash
railway logs
```

---

## What Happens Now

1. ✅ Your app is live on Railway
2. ✅ Free $5 monthly credit
3. ✅ Auto-scaling (if you get traffic)
4. ✅ 24/7 uptime
5. ✅ Automatic SSL/HTTPS
6. ✅ No infrastructure to manage

---

## Monitor Your App

Visit your Railway dashboard regularly to:
- Check resource usage
- View logs
- Update environment variables
- Scale services if needed
- Monitor costs

---

## After Deployment

Your app is now online! Share the URL with friends:

```
https://your-app-name.railway.app
```

---

## Further Reading

- [Railway Documentation](https://docs.railway.app)
- [Railway GitHub Integration](https://docs.railway.app/guides/github)
- [Railway Environment Variables](https://docs.railway.app/develop/variables)
- [Whisper API Docs](https://github.com/openai/whisper)

---

## Support

If you have issues:

1. Check Railway logs (click service → Logs)
2. Check browser console (F12 → Console tab)
3. Visit Railway docs: https://docs.railway.app
4. Check Flask logs for backend errors

---

**Congratulations! Your Farsi Transcriber is now live!** 🎉
