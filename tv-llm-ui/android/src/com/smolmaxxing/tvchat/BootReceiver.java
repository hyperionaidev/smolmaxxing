package com.smolmaxxing.tvchat;

import android.content.BroadcastReceiver;
import android.content.Context;
import android.content.Intent;

/** Starts the model server when the box boots (and after the app is updated). */
public class BootReceiver extends BroadcastReceiver {
    @Override
    public void onReceive(Context context, Intent intent) {
        ChatService.start(context);
    }
}
