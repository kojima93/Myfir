import json, os
from boxsdk import JWTAuth, Client

# GitHub Secrets から取得
cfg = json.loads(os.environ["BOX_CONFIG_JSON"])

# JWT 認証
auth = JWTAuth.from_settings_dictionary(cfg)

# 企業（サービスアカウント）でトークン発行
# ユーザー代行なら: auth.authenticate_user(os.environ["BOX_USER_ID"])
access_token = auth.authenticate_instance()

# 動作確認：自分（サービスアカウント）を取得
client = Client(auth)
me = client.user().get()
print(f"[OK] Box 認証成功: id={me.id}, name={me.name}")

# 下流ステップ用に保存（任意）
with open(".box_token", "w") as f:
    f.write(access_token)
