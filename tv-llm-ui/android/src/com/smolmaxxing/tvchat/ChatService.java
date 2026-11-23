package com.smolmaxxing.tvchat;

import android.app.Notification;
import android.app.NotificationChannel;
import android.app.NotificationManager;
import android.app.Service;
import android.content.Context;
import android.content.Intent;
import android.content.pm.ServiceInfo;
import android.os.IBinder;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.ArrayList;
import java.util.List;

/**
 * Foreground service that keeps llama-server running on 127.0.0.1:8090 under the tv-watch memory guard.
 * Both binaries ship in the APK as lib*.so (static aarch64 executables), so they live in the app's native
 * library directory, the one place an app may execute files from. The model is read from the app's external
 * files directory, where install.sh pushes it over adb.
 */
public class ChatService extends Service {
    static final String TAG = "SmolChat";
    static final int PORT = 8090;
    static final String MODEL = "q08-q4k-h65p8.gguf";   // Qwen3.5-0.8B: the 2B does not fit next to the page
    private static final String CHANNEL = "server";

    private volatile boolean running;
    private volatile Process proc;
    private Thread supervisor;

    static void start(Context c) {
        c.startForegroundService(new Intent(c, ChatService.class));
    }

    static File modelFile(Context c) {
        return new File(c.getExternalFilesDir(null), MODEL);
    }

    @Override
    public int onStartCommand(Intent intent, int flags, int startId) {
        NotificationManager nm = getSystemService(NotificationManager.class);
        nm.createNotificationChannel(new NotificationChannel(CHANNEL, "Model server", NotificationManager.IMPORTANCE_LOW));
        Notification n = new Notification.Builder(this, CHANNEL)
                .setContentTitle("SmolChat")
                .setContentText("Model server on 127.0.0.1:" + PORT)
                .setSmallIcon(android.R.drawable.stat_sys_download_done)
                .setOngoing(true)
                .build();
        startForeground(1, n, ServiceInfo.FOREGROUND_SERVICE_TYPE_SPECIAL_USE);
        if (supervisor == null) {
            running = true;
            supervisor = new Thread(this::supervise, "smolchat-supervisor");
            supervisor.start();
        }
        return START_STICKY;
    }

    /** Runs the server and restarts it if it exits; backs off when it keeps dying (e.g. the guard on low memory). */
    private void supervise() {
        int quickDeaths = 0;
        while (running) {
            long t0 = System.currentTimeMillis();
            try {
                File model = modelFile(this);
                if (!model.canRead()) {
                    Log.e(TAG, "model missing: " + model + " (push it with tv-llm-ui/android/install.sh)");
                    sleep(30_000);
                    continue;
                }
                killServers();   // a server left over from a crashed run would hold the port
                File ui = prepareUi();
                String lib = getApplicationInfo().nativeLibraryDir;
                List<String> cmd = new ArrayList<>();
                cmd.add(lib + "/libtvwatch.so");
                cmd.add(new File(getFilesDir(), "server.log").getPath());
                cmd.add(lib + "/libllamaserver.so");
                String[] args = {"-m", model.getPath(), "-t", "3", "-c", "1024", "-np", "1",
                        "--host", "0.0.0.0", "--port", String.valueOf(PORT), "--path", ui.getPath(),
                        "--metrics", "--reasoning", "off", "-ub", "128", "-b", "128"};
                for (String a : args) cmd.add(a);
                ProcessBuilder pb = new ProcessBuilder(cmd).directory(getFilesDir()).redirectErrorStream(true)
                        .redirectOutput(new File(getFilesDir(), "watch.log"));
                pb.environment().put("LLAMA_STATE_F16", "1");      // fp16 DeltaNet state, +11 % (EXPERIMENT.md §16)
                pb.environment().put("BOX_JSON", new File(ui, "box.json").getPath());
                pb.environment().put("LIMIT_MB", "120");           // the guard kills the server below this
                Log.i(TAG, "starting: " + cmd);
                proc = pb.start();
                int rc = proc.waitFor();
                Log.w(TAG, "server exited with " + rc);
            } catch (InterruptedException e) {
                break;
            } catch (Exception e) {
                Log.e(TAG, "server start failed", e);
            }
            if (!running) break;
            quickDeaths = System.currentTimeMillis() - t0 < 120_000 ? quickDeaths + 1 : 0;
            sleep(quickDeaths > 3 ? 300_000 : 10_000);
        }
    }

    /** Copies the page from the APK's assets into files/ui, where llama-server serves it with --path. */
    private File prepareUi() throws Exception {
        File ui = new File(getFilesDir(), "ui");
        ui.mkdirs();
        for (String name : getAssets().list("ui")) {
            try (InputStream in = getAssets().open("ui/" + name);
                 OutputStream out = new FileOutputStream(new File(ui, name))) {
                byte[] buf = new byte[16384];
                for (int r; (r = in.read(buf)) > 0; ) out.write(buf, 0, r);
            }
        }
        return ui;
    }

    /** Kills every llama-server of this app (same uid, so /proc/<pid>/cmdline is readable and the kill allowed). */
    private static void killServers() {
        File[] procs = new File("/proc").listFiles();
        if (procs == null) return;
        for (File d : procs) {
            String n = d.getName();
            if (!n.matches("\\d+")) continue;
            try (InputStream in = new java.io.FileInputStream(new File(d, "cmdline"))) {
                byte[] b = new byte[512];
                int r = in.read(b);
                if (r > 0 && new String(b, 0, r).contains("libllamaserver.so")) {
                    android.os.Process.sendSignal(Integer.parseInt(n), 9);
                    Log.i(TAG, "killed llama-server " + n);
                }
            } catch (Exception ignored) { }
        }
    }

    private void sleep(long ms) {
        try { Thread.sleep(ms); } catch (InterruptedException ignored) { }
    }

    @Override
    public void onDestroy() {
        running = false;
        Process p = proc;
        if (p != null) p.destroy();   // tv-watch does not forward SIGTERM, so stop llama-server ourselves too
        killServers();
        if (supervisor != null) supervisor.interrupt();
        super.onDestroy();
    }

    @Override
    public IBinder onBind(Intent intent) { return null; }
}
